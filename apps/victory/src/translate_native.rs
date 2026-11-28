//! Offline native translation. No device, pipeline or title is created. With `cache`, each
//! translation goes into the native shader cache `run` reads (the warm-up before a run), under
//! the options the environment selects: run it with the `VICTORY_*` variables of the run.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit;
use victory_shader::{Container, ShaderError};

use crate::shader::{MAX_FILE, sources};

pub fn run(paths: Vec<PathBuf>, list: bool, jobs: Option<usize>, cache: bool) -> Result<()> {
    let options = victory_gpu::native::shaders::options();
    println!("native options {options:?}");
    if cache {
        let cfg = victory_core::Config::load()?;
        let dir = victory_gpu::shader_cache::use_native_cache(cfg.cache_subdir("native-shaders")?);
        println!("native shader cache {dir:?}");
    }
    let jobs = jobs.unwrap_or(4).max(1);
    // `VICTORY_LIST_DXIL=1`: each listed translation with a hash of its DXIL, to compare the
    // output of two builds over the same containers.
    let list_dxil = std::env::var_os("VICTORY_LIST_DXIL").is_some();
    let seen = Mutex::new(HashSet::new());
    let outcomes = Mutex::new(Vec::new());
    for root in crate::default_paths(paths)? {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata()?.len() > MAX_FILE {
                continue;
            }
            let data = std::fs::read(entry.path())?;
            let sources = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..jobs.min(sources.len()) {
                    scope.spawn(|| {
                        while let Some(source) = sources.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = source.bytes() else {
                                continue;
                            };
                            if Container::parse(&bytes).is_err() {
                                continue;
                            }
                            // Raw sources extend to EOF. Hash only the checked container size,
                            // matching the runtime's logged shader identities.
                            let size =
                                u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
                            let bytes = &bytes[..size];
                            let hash = blake3::hash(bytes);
                            if !seen.lock().unwrap().insert(hash) {
                                continue;
                            }
                            let mut dxil = String::new();
                            let reason = if cache {
                                // The run's own entry point: translated once, kept on disk.
                                match victory_gpu::native::shaders::translate(bytes) {
                                    Ok(shader) => format!("translated {:?}", shader.stage),
                                    Err(e) => format!("refused: {e}"),
                                }
                            } else {
                                match emit::translate_with(bytes, options) {
                                    Ok(shader) => {
                                        if list_dxil {
                                            dxil = format!(" dxil {}", &blake3::hash(&shader.dxil).to_hex()[..16]);
                                        }
                                        format!("translated {:?}", shader.stage)
                                    }
                                    Err(ShaderError::Unsupported { what, .. }) => {
                                        format!("refused: {what}")
                                    }
                                    Err(e) => format!("refused: {e}"),
                                }
                            };
                            outcomes.lock().unwrap().push((
                                format!(
                                    "{} {}{}{dxil}",
                                    &hash.to_hex()[..16],
                                    entry.path().display(),
                                    source.label()
                                ),
                                reason,
                            ));
                        }
                    });
                }
            });
        }
    }
    let mut outcomes = outcomes.into_inner().unwrap();
    outcomes.sort();
    let mut counts = BTreeMap::<String, usize>::new();
    for (label, reason) in &outcomes {
        *counts.entry(reason.clone()).or_default() += 1;
        if list {
            println!("{label}  {reason}");
        }
    }
    println!("unique containers {}", outcomes.len());
    for (reason, count) in counts {
        println!("{count:6}  {reason}");
    }
    Ok(())
}

/// `translate --mesh-pairs`: links each listed mesh + pixel pair as the native host does, makes
/// the mesh program fit Metal's output limit, and prints what that took.
pub fn mesh_pairs(pairs: &std::path::Path, paths: Vec<PathBuf>) -> Result<()> {
    use victory_shader::emit::Fitted;
    let options = victory_gpu::native::shaders::options();
    let dir = paths.first().ok_or_else(|| anyhow::anyhow!("give the folder of the containers"))?;
    let out = std::env::var_os("VICTORY_MESH_PAIRS_OUT").map(PathBuf::from);
    if let Some(out) = &out {
        std::fs::create_dir_all(out)?;
    }
    let fit = victory_gpu::native::shaders::mesh_fit();
    let mut counts = BTreeMap::<&str, usize>::new();
    for line in std::fs::read_to_string(pairs)?.lines() {
        let mut words = line.split_whitespace();
        let (Some(mesh), Some(pixel)) = (words.next(), words.next()) else { continue };
        let ms = std::fs::read(dir.join(mesh))?;
        let ps = (pixel != "-").then(|| std::fs::read(dir.join(pixel))).transpose()?;
        let made = if fit {
            emit::translate_fitted(&ms, ps.as_deref(), options)
        } else {
            emit::translate_linked(&ms, ps.as_deref(), options)
                .map(|(m, pixel)| emit::FittedMesh { passes: vec![m], pixel, fit: Fitted::AsLinked })
        };
        let outcome = match made {
            Ok(m) => {
                let bytes = m.passes.iter().map(|p| emit::metal_mesh_bytes(&p.dxil)).max().flatten();
                let fits = bytes.is_some_and(|b| b <= emit::METAL_MESH_OUTPUT_BYTES);
                println!("{mesh} {pixel} bytes {bytes:?} {:?} {}", m.fit, emit::mesh_outputs_text(&m.passes[0].dxil));
                if let Some(out) = &out {
                    for (n, pass) in m.passes.iter().enumerate() {
                        let pass_name = if n == 0 { String::new() } else { n.to_string() };
                        std::fs::write(out.join(format!("{mesh}_{pixel}.mesh{pass_name}.dxil")), &pass.dxil)?;
                    }
                    if let Some(p) = &m.pixel {
                        std::fs::write(out.join(format!("{mesh}_{pixel}.pixel.dxil")), &p.dxil)?;
                    }
                }
                match (fits, m.fit) {
                    (false, _) => "over Metal's mesh output limit",
                    (true, Fitted::AsLinked) => "fit as linked",
                    (true, Fitted::Reordered { .. }) => "fit with their outputs reordered",
                    (true, _) => "fit split in passes",
                }
            }
            Err(e) => {
                println!("{mesh} {pixel} refused: {e}");
                "refused by the translator"
            }
        };
        *counts.entry(outcome).or_default() += 1;
    }
    println!("pairs {}", counts.values().sum::<usize>());
    for (outcome, count) in counts {
        println!("{count:6}  {outcome}");
    }
    Ok(())
}

/// `translate --sets`: makes the host pipeline of each listed shader set from the containers in
/// the given folder (a container's file name starts with its hash), as the native host would at
/// the set's first draw or dispatch, and prints what was made or the refusal.
#[cfg(target_os = "macos")]
pub fn sets(sets: &std::path::Path, paths: Vec<PathBuf>) -> Result<()> {
    let dir = paths.first().ok_or_else(|| anyhow::anyhow!("give the folder of the containers"))?;
    let files: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| Some(e.ok()?.path())).collect();
    let container = |hash: &str| -> Result<Vec<u8>> {
        let file = files
            .iter()
            .find(|f| f.file_name().is_some_and(|n| n.to_string_lossy().starts_with(hash)))
            .ok_or_else(|| anyhow::anyhow!("no container {hash}"))?;
        let bytes = std::fs::read(file)?;
        anyhow::ensure!(Container::parse(&bytes).is_ok(), "{hash} is not a container");
        let size = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        anyhow::ensure!(&blake3::hash(&bytes[..size]).to_hex()[..16] == hash, "{hash}: the file holds another container");
        Ok(bytes[..size].to_vec())
    };
    let mut offline = victory_gpu::native::offline::Offline::new().map_err(anyhow::Error::msg)?;
    let (mut made, mut refused) = (0, 0);
    for line in std::fs::read_to_string(sets)?.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        let stage = |name: &str| words.iter().position(|w| *w == name).and_then(|at| words.get(at + 1)).copied();
        let rect = words.contains(&"rect");
        let name = words.iter().filter(|w| **w != "rect").copied().collect::<Vec<_>>().join(" ");
        let outcome = match (stage("vertex"), stage("pixel"), stage("compute")) {
            (Some(vertex), pixel, None) => match (container(vertex), pixel.map(container).transpose()) {
                (Ok(vs), Ok(ps)) => offline.graphics(&vs, ps.as_deref(), rect, &name),
                (Err(e), _) | (_, Err(e)) => Err(e.to_string()),
            },
            (None, None, Some(compute)) => match container(compute) {
                Ok(cs) => offline.compute(&cs, &name),
                Err(e) => Err(e.to_string()),
            },
            _ => continue,
        };
        let rect = if rect { " as rectangles" } else { "" };
        match outcome {
            Ok(what) => {
                made += 1;
                println!("{name}{rect}: pipeline made: {what}");
            }
            Err(why) => {
                refused += 1;
                println!("{name}{rect}: refused: {why}");
            }
        }
    }
    println!("sets {}: {made} made, {refused} refused", made + refused);
    Ok(())
}
