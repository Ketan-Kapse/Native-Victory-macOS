# The opening cutscene's outfit and black faces: textures the GPU makes at run time (28 November 2026)

Follows "The opening cutscene's outfit" in `native-host.md`. The fault: in the diner cutscene
Lucia wears the prison top (same garment, only the albedo differs), a waitress's head is black
with its outlines visible, some materials come out exactly black from run to run. NOT FIXED: the
switches below are off by default and no game run has judged them yet.

## What the title does (run logs, `VICTORY_TRACE_TARGETS` pipeline binds and copy kinds)

Shader names from the title's effect list; frame numbers from runs O1 and an earlier
native-merge run (the same order in both).

1. Skin: `ped_standard_skin_tattoo_bake` (`MSNC_VS_BakeDecorations` mesh shader +
   `PS_BakeDecorations_0000`) at frame ~1250, then the compute passes `CS_DecorationCopy`,
   `CS_DecorationDilate`, then `texturecompression` `CS_CompressBC3Mip0`, then
   `CopyTextureRegion` RGBA32_UINT -> BC3 sRGB (the first of its kind, the same second).
2. `decoration_blend` (`PS_DecorationBlend_0000`, `_0011`) at frame ~1466-1475, then
   `CS_CompressBC1Mip0`, then `CopyTextureRegion` RG32_UINT -> BC1.
3. At the cutscene's start (frame ~1657, about +95 s): `PS_DecorationBlend_1010`, then
   `CS_CompressBC7Mip0`, then `CopyTextureRegion` RGBA32_UINT -> BC7 sRGB; then RGBA32_UINT -> BC3
   and more `PS_DecorationBlend_0010` / `_1000` draws.

So characters' skin and clothes are composed on the GPU at run time (decals, tattoos, garment
variants drawn into a target), compressed by a compute pass that writes one RGBA32_UINT /
RG32_UINT texel per block, and copied into a block-compressed texture object. The compressors
are the `Mip0` variants: the copies write level 0.

## What our host did with it

* `VICTORY_TEXTURE_DROP_TOP_MIP=1` (on in every run of the user's) makes a sampled 2D texture of
  1024 texels or more with a lower level WITHOUT its top level. The encoders' destinations can be
  such textures (sampled only, never targets). A `CopyTextureRegion` into level 0 of one had no
  host level to go to and was skipped without a word (the copy paths in `glue.rs`), so the texture kept what
  title memory had: zeros for a fresh object (a black face with the mesh's outlines lit), or the
  bytes of what was there before (the base garment). Now logged by default: "a GPU copy names a
  level the host texture was made without" (first 16, then powers of two). NOT YET SEEN IN A
  RUN: whether the destinations are that large is the first thing the next run shows.
* A placed view of the destination's memory has a copy of its own filled from title memory,
  which never receives the GPU's result (`Title::alias_placed` shared only R9G9B9E5 views of
  R32_UINT objects).
* A decoration bake reads its sources through heap indices in its constants, no table: the host
  cannot tell which entries it reads, and covered-clear (`VICTORY_COVERED_CLEAR`, default on) or
  the texture budget may have left those entries null (exact black in O1).

## The switches (each `=1`; read once; logged as "texture switches on")

* `VICTORY_COPY_FULL` (ON by default since run T1, 2026-11-27: faces with eyes and hair; `=0`
  turns it off): a GPU copy to or from a level its host texture was made without gives
  that texture every level (`Title::full_levels`): a new host texture under every key that
  showed the old one, entries written again, filled from title memory at the next submission,
  before the copy runs. Not for a texture the GPU wrote already (its levels are not in title
  memory): that copy is still skipped and logged.
* `VICTORY_PLACED_SHARE`: a placed view over the memory of a texture the GPU writes shows that
  texture when one host texture can show both (same format or its sRGB partner; Metal aborts on
  a block-compressed view of another compression, as pragya's #499 found on D3DMetal), whichever
  of the two the host met first. A block-compressed view of plain texels of the block's size
  (RGBA32_UINT under BC7, the D3D12 path's `resource::alias_source` case) cannot share: the
  object's texture is copied into the view's, level by level while their sizes in blocks agree,
  in a list of the host's own run after the lists of each submission within three of a write to
  the object (`Title::alias_copy_list`); the view is not filled from title memory after its first
  upload. Logged: "a placed view shares the texture the GPU writes at its memory", "a placed
  block-compressed view over plain texels the GPU writes".
* `VICTORY_BAKE_SOURCES`: before a draw whose shaders take views from the heap at indices they
  compute (`HostShader::heap`), every word of its root constants and of the first 4 KiB of each
  root constant buffer that is an index of a texture record in the bound heap is made sure of:
  a texture missing on the host (covered, evicted, never made) is made again and its entry
  written again, before the draw is recorded; its contents are uploaded at the submission that
  runs the draw, before the GPU does (`textures/bake.rs`). Such textures are kept from
  covered-clear and the budget for 64 submissions after a draw named them (`Title::heap_read`).
  Guess: that a constant word below the heap's size is an index; a word that only looks like one
  keeps one texture more, an index computed otherwise is not seen. Logged loudly: "a heap-indexed
  draw names a texture its entry did not show" (with the pipeline's shader hashes, index, key,
  covered / evicted), "... not uploaded yet", and a summary at power-of-two draw counts with the
  time spent (`ms`).
* `VICTORY_REMIRROR`: a texture made again is shown again by every entry that showed it.

`VICTORY_TRACE_TARGETS=1` also logs the first 256 copies between plain texels and compressed
blocks: both textures (size, levels), title and host levels (`None`: skipped), the destination's
memory and whether a placed view there shares its host texture.

## Judged against other work (read only)

* Upstream PR #428 (froziiik: epochs for covered copies, overlap by pool pages, identical
  textures share by pool offset, top levels upgraded later) is not a better base for this fault:
  it also clears the entries of cold heap-shown copies once their last table-named submission
  retired, and heap-indexed reads name no table. Its sharing of identical textures by pool
  offset would lower memory under any rule.
* `VICTORY_PLACED_AT_ONCE=keep` (pragya #539) keeps a rewritten slot's old view until the new
  texture is ready. The native host makes and registers a placed texture when the record is
  mirrored and uploads it before the work that reads it runs, so it has no such gap; its nulls
  come from covered-clear and the budget, which `VICTORY_BAKE_SOURCES` answers for the draws it
  sees.

## Tests (synthetic texels)

`texture_tests::an_encoder_copy_into_a_level_left_off_gives_the_texture_every_level`,
`a_placed_compressed_view_over_plain_texels_the_gpu_writes_gets_them_copied`,
`a_placed_srgb_view_over_a_gpu_written_compressed_object_shares_its_texture`,
`a_heap_indexed_draw_gets_back_a_covered_texture_its_constants_name`.
