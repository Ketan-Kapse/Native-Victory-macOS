//! Converts a mesh program and its pixel program (DXIL, for example the linked pair
//! `victory translate --mesh-pairs` writes) with the Metal Shader Converter, outside any run.
//!
//! usage: mesh_pair <mesh.dxil> [<pixel.dxil>|-] [--dump <prefix>]
//!
//! Without `--dump` the pair is built as a mesh pipeline through the backend, as a run would
//! (colour targets chosen from the pixel program's output types). With it, `<prefix>.txt`,
//! `<prefix>.mesh.metallib` and `<prefix>.pixel.metallib` are written, the files of
//! `VICTORY_DUMP_MESH_PAIRS`, for a tool that hands them to Metal directly.

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use victory_metal::resources::{Cull, Format, GraphicsBackend, MeshPipelineDesc, RenderPipelineDesc};
    use victory_shader::emit::{METAL_MESH_OUTPUT_BYTES, mesh_outputs_text, metal_mesh_bytes};
    use victory_shader::{Container, FourCC};

    let args: Vec<String> = std::env::args().skip(1).collect();
    let dump = args.iter().position(|a| a == "--dump").map(|i| args[i + 1].clone());
    let files: Vec<&String> = args.iter().take_while(|a| *a != "--dump").collect();
    let mesh = std::fs::read(files[0]).expect("the mesh DXIL");
    let pixel = files.get(1).filter(|p| p.as_str() != "-").map(|p| std::fs::read(p).expect("the pixel DXIL"));
    let bytes = metal_mesh_bytes(&mesh);
    // A float target for a float output, an integer one for an integer output.
    let colors: Vec<Format> = pixel.as_deref().map_or_else(Vec::new, |ps| {
        let c = Container::parse(ps).expect("a pixel container");
        let part = c.part(FourCC::OSG1).map_or(&[][..], |p| p.data);
        let word = |at: usize| part.get(at..at + 4).map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()));
        let mut targets: Vec<(u32, u32)> = (0..word(0) as usize)
            .map(|i| 8 + 32 * i)
            .filter(|&at| word(at + 12) == 64)
            .map(|at| (word(at + 8), word(at + 16)))
            .collect();
        targets.sort();
        targets.iter().map(|t| if t.1 == 3 { Format::Rgba16Float } else { Format::Rgba32Uint }).collect()
    });
    if let Some(prefix) = dump {
        let msc = victory_metal::msc::Msc::open_default();
        let (lib, stage) = msc.convert_mesh_with(&mesh, None, false).expect("mesh conversion");
        std::fs::write(format!("{prefix}.mesh.metallib"), lib).unwrap();
        if let Some(ps) = &pixel {
            let (_, lib) = msc.convert_with(ps, None, None, false).expect("pixel conversion");
            std::fs::write(format!("{prefix}.pixel.metallib"), lib).unwrap();
        }
        let formats: Vec<&str> = colors.iter().map(|f| if *f == Format::Rgba16Float { "115" } else { "123" }).collect();
        let meta = format!(
            "entry {}\nthreads {}\ncolors {}\ndepth 252\nstencil 0\nbytes {}\nvertices {}\nprimitives {}\n",
            stage.entry,
            stage.info.threads.iter().product::<u32>(),
            formats.join(","),
            bytes.unwrap_or(0),
            stage.info.max_vertices,
            stage.info.max_primitives,
        );
        std::fs::write(format!("{prefix}.txt"), meta).unwrap();
        println!("dumped {prefix}: {bytes:?} bytes; {}", mesh_outputs_text(&mesh));
        return;
    }
    let m = victory_metal::metal::Metal::new().expect("a Metal device");
    let made = m.create_render_pipeline(&RenderPipelineDesc {
        mesh: Some(MeshPipelineDesc { mesh: &mesh, layout: None, amplification: None, amplification_layout: None, more: &[] }),
        ps: pixel.as_deref(),
        color: &colors,
        depth: Some(Format::D32Float),
        cull: Cull::None,
        ..Default::default()
    });
    println!(
        "{} bytes {bytes:?} of {METAL_MESH_OUTPUT_BYTES}: {}",
        files[0],
        match made {
            Ok(_) => "pipeline built".to_owned(),
            Err(e) => format!("FAILED: {e}"),
        }
    );
}
