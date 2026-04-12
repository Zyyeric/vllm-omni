import argparse
import time

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference script for MagiHuman.")
    parser.add_argument("--model", type=str, required=True, help="Path or ID of the MagiHuman model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt containing visual description, dialogue, and background sound.",
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp", type=int, default=4, help="Tensor parallel size (number of GPUs)."
    )
    parser.add_argument(
        "--output", type=str, default="output_magihuman.mp4", help="Path to save the generated mp4 file."
    )
    parser.add_argument("--height", type=int, default=256, help="Video height.")
    parser.add_argument("--width", type=int, default=448, help="Video width.")
    parser.add_argument("--seconds", type=int, default=5, help="Generated video duration in seconds.")
    parser.add_argument("--disable-sr", action="store_true", help="Disable the super-resolution stage.")
    parser.add_argument("--sr-height", type=int, default=1080, help="Super-resolution output height.")
    parser.add_argument("--sr-width", type=int, default=1920, help="Super-resolution output width.")
    parser.add_argument("--sr-num-inference-steps", type=int, default=5, help="Super-resolution denoising steps.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=52, help="Random seed for generation.")
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit"],
        help="Cache backend for acceleration. Default: None.",
    )
    parser.add_argument(
        "--enable-cache-dit-summary",
        action="store_true",
        help="Enable cache-dit summary logging after diffusion forward passes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "max_cached_steps": 20,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }

    print(f"Initializing MagiHuman pipeline with TP={args.tensor_parallel_size}...")
    init_start = time.perf_counter()
    omni = Omni(
        model=args.model,
        init_timeout=1200,
        tensor_parallel_size=args.tensor_parallel_size,
        devices=list(range(args.tensor_parallel_size)),
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_cache_dit_summary=args.enable_cache_dit_summary,
    )
    init_time = time.perf_counter() - init_start
    print(f"Initialization time: {init_time:.2f}s")

    prompt = args.prompt
    if not prompt:
        prompt = (
            "A young woman with long, wavy golden blonde hair and bright blue eyes, "
            "wearing a fitted ivory silk blouse with a delicate lace collar, sits "
            "stationary in front of a softly lit, blurred warm-toned interior. Her "
            "overall disposition is warm, composed, and gently confident. The camera "
            "holds a static medium close-up, framing her from the shoulders up, "
            "with shallow depth of field keeping her face in sharp focus. Soft "
            "directional key light falls from the upper left, casting a gentle "
            "highlight along her cheekbone and nose bridge. She draws a quiet breath, "
            "the levator labii superiors relaxing as her lips part. She speaks in "
            "clear, warm, unhurried American English: "
            "\"The most beautiful things in life aren't things at all — "
            "they're moments, feelings, and the people who make you feel truly alive.\" "
            "Her jaw descends smoothly on each stressed syllable; the orbicularis oris "
            "shapes each vowel with precision. A faint, genuine smile engages the "
            "zygomaticus major, lifting her lip corners fractionally. Her brows rest "
            "in a soft, neutral arch throughout. She maintains steady, forward-facing "
            "eye contact. Head position remains level; no torso displacement occurs.\n\n"
            "Dialogue:\n"
            "<Young blonde woman, American English>: "
            "\"The most beautiful things in life aren't things at all — "
            "they're moments, feelings, and the people who make you feel truly alive.\"\n\n"
            "Background Sound:\n"
            "<Soft, warm indoor ambience with a faint distant piano melody>"
        )

    extra_args = {"seconds": args.seconds}
    if not args.disable_sr:
        extra_args.update(
            {
                "sr_height": args.sr_height,
                "sr_width": args.sr_width,
                "sr_num_inference_steps": args.sr_num_inference_steps,
            }
        )

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args=extra_args,
    )

    print(f"Cache backend: {args.cache_backend or 'None (no acceleration)'}")
    print(f"Generating with prompt: {prompt[:80]}...")
    generation_start = time.perf_counter()
    outputs = omni.generate(
        prompts=[prompt],
        sampling_params_list=[sampling_params],
    )
    generation_time = time.perf_counter() - generation_start
    print(f"Generation time (generate only): {generation_time:.2f}s")

    print(f"Generation complete. Output type: {type(outputs)}")
    if outputs:
        first = outputs[0]

        if hasattr(first, "images") and first.images:
            video_frames = first.images[0]
            print(f"Video frames: shape={video_frames.shape}, dtype={video_frames.dtype}")

            audio_waveform = None
            mm = first.multimodal_output or {}
            if mm:
                audio_waveform = mm.get("audio")
                if audio_waveform is not None:
                    print(f"Audio waveform: shape={audio_waveform.shape}, dtype={audio_waveform.dtype}")

            output_fps = float(mm.get("fps", 25))
            output_sr = int(mm.get("audio_sample_rate", 24000))
            print(f"Using fps={output_fps}, audio_sample_rate={output_sr} from model output")

            video_bytes = mux_video_audio_bytes(
                video_frames,
                audio_waveform,
                fps=output_fps,
                audio_sample_rate=output_sr,
            )
            with open(args.output, "wb") as f:
                f.write(video_bytes)
            print(f"Saved MP4 ({len(video_bytes)} bytes) to {args.output}")
        print("SUCCESS: MagiHuman pipeline generation completed.")
    else:
        print("WARNING: No outputs returned.")


if __name__ == "__main__":
    main()
