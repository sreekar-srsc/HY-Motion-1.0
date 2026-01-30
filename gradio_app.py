import argparse
import codecs as cs
import json
import os
import os.path as osp
import random
import re
import textwrap
from typing import List, Optional, Tuple, Union

import glob
import gradio as gr
from hymotion.utils.gradio_runtime import ModelInference
from hymotion.utils.gradio_utils import try_to_download_model, try_to_download_text_encoder
from hymotion.utils.gradio_css import get_placeholder_html, APP_CSS, HEADER_BASE_MD, FOOTER_MD, WITHOUT_PROMPT_ENGINEERING_WARNING
from hymotion.utils.visualize_mesh_web import get_cached_smpl_frames
# Import spaces for Hugging Face Zero GPU support
import spaces

# define data sources
DATA_SOURCES = {
    "example_prompts": "examples/example_prompts/example_subset.json",
}

# Pre-generated examples for gallery display (generated on first startup)
# Add/remove items to control the number of examples
EXAMPLE_GALLERY_LIST = [
    {
        "prompt": "A person jumps upward with both legs twice.",
        "duration": 4.5,
        "seeds": "792",
        "cfg_scale": 5.0,
        "filename": "jump_twice",
    },
    # Add more examples here as needed:
    {
        "prompt": "A person jumps on their right leg.",
        "duration": 4.5,
        "seeds": "941",
        "cfg_scale": 5.0,
        "filename": "jump_right_leg",
    },
]
EXAMPLE_GALLERY_OUTPUT_DIR = "examples/pregenerated"

def ensure_examples_generated(model_inference_obj) -> List[str]:
    """
    Ensure all example motions are generated on first startup.
    Returns a list of successfully generated example filenames.
    """
    example_dir = EXAMPLE_GALLERY_OUTPUT_DIR
    os.makedirs(example_dir, exist_ok=True)

    generated_examples = []

    for example in EXAMPLE_GALLERY_LIST:
        example_filename = example["filename"]
        meta_path = os.path.join(example_dir, f"{example_filename}_meta.json")

        # Check if already generated
        if os.path.exists(meta_path):
            print(f">>> Example already exists: {meta_path}")
            generated_examples.append(example_filename)
            continue

        # Generate the example
        print(f">>> Generating example motion: {example['prompt']}")
        try:
            # Force CPU device for example generation at startup
            # This is necessary for Hugging Face Zero GPU environment where GPU
            # is only available inside @spaces.GPU decorated functions
            html_content, fbx_files = model_inference_obj.run_inference(
                text=example["prompt"],
                seeds_csv=example["seeds"],
                motion_duration=example["duration"],
                cfg_scale=example["cfg_scale"],
                output_format="dict",  # Don't generate FBX for example
                original_text=example["prompt"],
                output_dir=example_dir,
                output_filename=example_filename,
                device="cpu",  # Force CPU for startup example generation
            )
            print(f">>> Example '{example_filename}' generated successfully!")
            generated_examples.append(example_filename)
        except Exception as e:
            print(f">>> Failed to generate example '{example_filename}': {e}")

    return generated_examples


def load_example_gallery_html(example_index: int = 0) -> str:
    """
    Load a specific pre-generated example and return iframe HTML for display.
    Args:
        example_index: Index of the example in EXAMPLE_GALLERY_LIST
    """
    from hymotion.utils.visualize_mesh_web import generate_static_html_content

    if example_index < 0 or example_index >= len(EXAMPLE_GALLERY_LIST):
        return ""

    example = EXAMPLE_GALLERY_LIST[example_index]
    example_dir = EXAMPLE_GALLERY_OUTPUT_DIR
    example_filename = example["filename"]
    meta_path = os.path.join(example_dir, f"{example_filename}_meta.json")

    if not os.path.exists(meta_path):
        return f"""
        <div style='height: 300px; display: flex; justify-content: center; align-items: center;
                    background: #2d3748; border-radius: 12px; color: #a0aec0;'>
            <p>Example not generated yet. Please restart the app.</p>
        </div>
        """

    try:
        html_content = generate_static_html_content(
            folder_name=example_dir,
            file_name=example_filename,
            hide_captions=False,
        )
        escaped_html = html_content.replace('"', "&quot;")
        iframe_html = f"""
            <iframe
                srcdoc="{escaped_html}"
                width="100%"
                height="350px"
                style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
            ></iframe>
        """
        return iframe_html
    except Exception as e:
        print(f">>> Failed to load example gallery: {e}")
        return ""


def get_example_gallery_grid_html() -> str:
    """
    Generate a grid layout HTML for all examples in the gallery.
    """
    if not EXAMPLE_GALLERY_LIST:
        return "<p>No examples configured.</p>"

    # Calculate grid columns based on number of examples
    num_examples = len(EXAMPLE_GALLERY_LIST)
    if num_examples == 1:
        columns = 1
    elif num_examples == 2:
        columns = 2
    elif num_examples <= 4:
        columns = 2
    else:
        columns = 3

    grid_items = []
    for idx, example in enumerate(EXAMPLE_GALLERY_LIST):
        iframe_html = load_example_gallery_html(idx)
        prompt_short = example["prompt"][:60] + "..." if len(example["prompt"]) > 60 else example["prompt"]

        grid_items.append(f"""
            <div class="example-grid-item" style="background: var(--card-bg, #fff); border-radius: 12px;
                        padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 14px; font-weight: 600; color: var(--text-primary, #333);
                            margin-bottom: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {prompt_short}
                </div>
                {iframe_html}
            </div>
        """)

    grid_html = f"""
        <div style="display: grid; grid-template-columns: repeat({columns}, 1fr); gap: 16px; padding: 8px;">
            {"".join(grid_items)}
        </div>
    """
    return grid_html


def load_examples_from_txt(txt_path: str, example_record_fps=20, max_duration=12):
    """Load examples from txt file."""

    def _parse_line(line: str) -> Optional[Tuple[str, float]]:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split("#")
            if len(parts) >= 2:
                text = parts[0].strip()
                duration = int(parts[1]) / example_record_fps
                duration = min(duration, max_duration)
            else:
                text = line.strip()
                duration = 5.0
            return text, duration
        return None

    examples: List[Tuple[str, float]] = []
    if os.path.exists(txt_path):
        try:
            if txt_path.endswith(".txt"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        result = _parse_line(line)
                        if result is None:
                            continue
                        text, duration = result
                        examples.append((text, duration))
            elif txt_path.endswith(".json"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = json.load(f)
                    for key, value in lines.items():
                        if "_raw_chn" in key or "GENERATE_PROMPT_FORMAT" in key:
                            continue
                        for line in value:
                            result = _parse_line(line)
                            if result is None:
                                continue
                            text, duration = result
                            examples.append((text, duration))
            print(f">>> Loaded {len(examples)} examples from {txt_path}")
        except Exception as e:
            print(f">>> Failed to load examples from {txt_path}: {e}")
    else:
        print(f">>> Examples file not found: {txt_path}")

    return examples


@spaces.GPU(duration=120)  # Request GPU for up to 120 seconds per inference
def generate_motion_func(
    # text input
    original_text: str,
    rewritten_text: str,
    # model input
    seed_input: str,
    motion_duration: float,
    cfg_scale: float,
) -> Tuple[str, List[str]]:
    use_prompt_engineering = USE_PROMPT_ENGINEERING
    output_dir = "output/gradio"
    # Determine which text to use: prefer rewritten_text, fallback to original_text
    if use_prompt_engineering and rewritten_text.strip():
        text_to_use = rewritten_text.strip()
    elif original_text.strip():
        text_to_use = original_text.strip()
    else:
        # Both are empty
        return "Error: Input text is empty, please enter text first", []

    try:
        # Use runtime from global if available (for Zero GPU), otherwise use self.runtime
        fbx_ok = model_inference.fbx_available
        req_format = "fbx" if fbx_ok else "dict"

        # Use GPU-decorated wrapper function for Zero GPU support
        # This ensures the GPU decorator receives proper Gradio context for user authentication
        html_content, fbx_files = model_inference.run_inference(
            text=text_to_use,
            seeds_csv=seed_input,
            motion_duration=motion_duration,
            cfg_scale=cfg_scale,
            output_format=req_format,
            original_text=original_text,
            output_dir=output_dir,
        )
        print(f"Running inference...after gpu_inference_wrapper")
        # Escape HTML content for srcdoc attribute
        escaped_html = html_content.replace('"', "&quot;")
        # Return iframe with srcdoc - directly embed HTML content
        iframe_html = f"""
            <iframe
                srcdoc="{escaped_html}"
                width="100%"
                height="750px"
                style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
            ></iframe>
        """
        return iframe_html, fbx_files
    except Exception as e:
        print(f"\t>>> Motion generation failed: {e}")
        return (
            f"❌ Motion generation failed: {str(e)}\n\nPlease check the input parameters or try again later",
            [],
        )


@spaces.GPU(duration=120)
def generate_smpl_frames_on_gpu(
    text: str,
    duration: float,
    seeds_csv: str,
    cfg_scale: float,
) -> list:
    """Generate motion and return SMPL frames as JSON for VSCode extension."""
    output_dir = "output/gradio"

    model_inference.run_inference(
        text=text,
        seeds_csv=seeds_csv,
        motion_duration=duration,
        cfg_scale=cfg_scale,
        output_format="dict",
        original_text=text,
        output_dir=output_dir,
    )

    # Find and return frames from saved NPZ files
    meta_files = sorted(
        glob.glob(os.path.join(output_dir, "*_meta.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if meta_files:
        base_filename = os.path.basename(meta_files[0]).replace("_meta.json", "")
        return get_cached_smpl_frames(output_dir, base_filename)
    return []


def smpl_frames_api(
    text: str,
    duration: float = 5.0,
    seeds: str = "0,1,2,3",
    cfg_scale: float = 5.0,
) -> dict:
    """API endpoint for VSCode extension - returns SMPL frame data."""
    try:
        if not text or not text.strip():
            return {"status": "error", "error": "Text required", "frames": [], "metadata": {}}

        duration = max(0.5, min(12.0, float(duration)))
        cfg_scale = max(1.0, min(10.0, float(cfg_scale)))
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()] or [0, 1, 2, 3]

        frames = generate_smpl_frames_on_gpu(
            text.strip(),
            duration,
            ",".join(str(s) for s in seed_list[:4]),
            cfg_scale,
        )

        if not frames:
            return {"status": "error", "error": "No frames generated", "frames": [], "metadata": {}}

        return {
            "status": "success",
            "frames": frames,
            "metadata": {
                "prompt": text.strip(),
                "duration": duration,
                "fps": 30,
                "total_frames": len(frames),
                "seeds": seed_list[:4],
                "cfg_scale": cfg_scale,
            },
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)}", "frames": [], "metadata": {}}


class T2MGradioUI:
    def __init__(self, args):
        self.output_dir = args.output_dir
        print(f"[{self.__class__.__name__}] output_dir: {self.output_dir}")
        # self.args = args
        self.prompt_engineering_available = args.use_prompt_engineering
        if self.prompt_engineering_available:
            try:
                from hymotion.prompt_engineering.client import PromptEngineeringClient
                self.prompt_engineering_client = PromptEngineeringClient()
                # Test the client with a simple prompt to verify it works
                self.prompt_engineering_client.rewrite_prompt_and_infer_time("A person walks forward.", max_timeout=30)
                print(f"[{self.__class__.__name__}] Prompt engineering client initialized successfully.")
            except Exception as e:
                print(f"[{self.__class__.__name__}] Prompt engineering client initialization failed: {e}")
                self.prompt_engineering_available = False
                # IMPORTANT: Update global variable so generate_motion_func uses correct behavior
                global USE_PROMPT_ENGINEERING
                USE_PROMPT_ENGINEERING = False
                print(f"[{self.__class__.__name__}] USE_PROMPT_ENGINEERING set to False due to initialization failure")


        self.all_example_data = {}
        self._init_example_data()

    def _init_example_data(self):
        for source_name, file_path in DATA_SOURCES.items():
            examples = load_examples_from_txt(file_path)
            if examples:
                self.all_example_data[source_name] = examples
            else:
                # provide default examples as fallback
                self.all_example_data[source_name] = [
                    ("Twist at the waist and punch across the body.", 3.0),
                    ("A person is running then takes big leap.", 3.0),
                    ("A person holds a railing and walks down a set of stairs.", 5.0),
                    (
                        "A man performs a fluid and rhythmic hip-hop style dance, incorporating body waves, arm gestures, and side steps.",
                        5.0,
                    ),
                ]
        print(f">>> Loaded data sources: {list(self.all_example_data.keys())}")

    def _get_header_text(self):
        return HEADER_BASE_MD

    def _generate_random_seeds(self):
        seeds = [random.randint(0, 999) for _ in range(4)]
        return ",".join(map(str, seeds))

    def _prompt_engineering(self, text: str, duration: float):
        if not text.strip():
            return "", gr.update(interactive=False), gr.update(), "⚠️ Please enter text first"

        print(f"\t>>> Using LLM to estimate duration/rewrite text...")
        try:
            predicted_duration, rewritten_text = self.prompt_engineering_client.rewrite_prompt_and_infer_time(text=text)
        except Exception as e:
            print(f"\t>>> Text rewriting/duration prediction failed: {e}")
            # On failure, use original text and enable generate button
            return (
                text,  # Use original text as fallback
                gr.update(interactive=True),  # Enable generate button
                gr.update(),
                f"⚠️ Text rewriting failed: {str(e)}\n💡 Using your original input directly. You can click [🚀 Generate Motion] to continue.",
            )

        return (
            rewritten_text,
            gr.update(interactive=True),
            gr.update(value=predicted_duration),
            "✅ Text rewriting completed! Please check and edit the rewritten text, then click [🚀 Generate Motion]",
        )

    def _get_example_choices(self):
        """Get all example choices from all data sources"""
        choices = ["Custom Input"]
        for source_name in self.all_example_data:
            example_data = self.all_example_data[source_name]
            for text, _ in example_data:
                display_text = f"{text[:50]}..." if len(text) > 50 else text
                choices.append(display_text)
        return choices

    def _on_example_select(self, selected_example):
        """When selecting an example, the callback function"""
        if selected_example == "Custom Input":
            if self.prompt_engineering_available:
                return "", self._generate_random_seeds(), gr.update(), gr.update(value="", visible=False), gr.update(interactive=False), "Please enter text or select an example"
            else:
                return "", self._generate_random_seeds(), gr.update(), gr.update(), gr.update(), gr.update()
        else:
            # find the corresponding example from all data sources
            for source_name in self.all_example_data:
                example_data = self.all_example_data[source_name]
                for text, duration in example_data:
                    display_text = f"{text[:50]}..." if len(text) > 50 else text
                    if display_text == selected_example:
                        if self.prompt_engineering_available:
                            # Set text directly to rewritten_text and enable generate button
                            return text, self._generate_random_seeds(), gr.update(value=duration), gr.update(value=text, visible=True), gr.update(interactive=True), "✅ Example selected! Click [🚀 Generate Motion] to start."
                        else:
                            return text, self._generate_random_seeds(), gr.update(value=duration), gr.update(), gr.update(), gr.update()
            if self.prompt_engineering_available:
                return "", self._generate_random_seeds(), gr.update(), gr.update(value="", visible=False), gr.update(interactive=False), "Please enter text or select an example"
            else:
                return "", self._generate_random_seeds(), gr.update(), gr.update(), gr.update(), gr.update()

    def build_ui(self):
        with gr.Blocks(css=APP_CSS) as demo:
            # Create State components for non-UI values that need to be passed to event handlers
            self.use_prompt_engineering_state = gr.State(self.prompt_engineering_available)
            self.output_dir_state = gr.State(self.output_dir)

            self.header_md = gr.Markdown(HEADER_BASE_MD, elem_classes=["main-header"])

            with gr.Row():
                # Left control panel
                with gr.Column(scale=2, elem_classes=["left-panel"]):

                    # Input textbox
                    if self.prompt_engineering_available:
                        input_place_holder = "Enter text to generate motion, support Chinese and English text input. Non-humanoid Characters, Multi-person Interactions and Environment & Camera are not supported. Click [ 📚 Example Prompts ] to see more examples."
                    else:
                        input_place_holder = "Enter English text to generate motion, please use `A person ...` format to describe the motion, better less than 50 words. Non-humanoid Characters, Multi-person Interactions and Environment & Camera are not supported. Click [ 📚 Example Prompts ] to see more examples."

                    self.text_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder=input_place_holder,
                        lines=3,
                        max_lines=10,
                        autoscroll=False,
                    )
                    # if not self.prompt_engineering_available:
                    #     gr.Markdown(
                    #         "Click [📚 Example Prompts] to see more examples."
                    #     )
                    # Rewritten textbox
                    self.rewritten_text = gr.Textbox(
                        label="✏️ Rewritten Text",
                        placeholder="Rewritten text will be displayed here, you can further edit",
                        interactive=True,
                        visible=False,
                    )
                    # Duration slider
                    self.duration_slider = gr.Slider(
                        minimum=0.5,
                        maximum=12,
                        value=5.0,
                        step=0.1,
                        label="⏱️ Action Duration (seconds)",
                        info="Feel free to adjust the action duration",
                    )

                    # Execute buttons
                    with gr.Row():
                        if self.prompt_engineering_available:
                            self.rewrite_btn = gr.Button(
                                "🔄 Rewrite Text",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                            )
                        else:
                            # Create a hidden/disabled placeholder button
                            self.rewrite_btn = gr.Button(
                                "🔄 Rewrite Text (Unavailable)",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                                interactive=False,
                                visible=False,
                            )

                        self.generate_btn = gr.Button(
                            "🚀 Generate Motion",
                            variant="primary",
                            size="lg",
                            elem_classes=["generate-button"],
                            interactive=not self.prompt_engineering_available,  # Enable directly if rewrite not available
                        )


                    # Example selection dropdown
                    self.example_dropdown = gr.Dropdown(
                        choices=self._get_example_choices(),
                        value="Custom Input",
                        label="📚 Example Prompts",
                        # info="Select a preset example or input your own text above",
                        interactive=True,
                    )

                    # Advanced settings
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        self._build_advanced_settings()

                    # Status message depends on whether rewrite is available
                    if self.prompt_engineering_available:
                        status_msg = "Please click the [🔄 Rewrite Text] button to rewrite the text first"
                    else:
                        status_msg = "Enter your text and click [🚀 Generate Motion] directly."

                    self.status_output = gr.Textbox(
                        label="📊 Status Information",
                        value=status_msg,
                        lines=1,
                        max_lines=10,
                        elem_classes=["status-textbox"],
                    )

                    # FBX Download section
                    with gr.Row(visible=False) as self.fbx_download_row:
                        if model_inference.fbx_available:
                            self.fbx_files = gr.File(
                                label="📦 Download FBX Files",
                                file_count="multiple",
                                interactive=False,
                            )
                        else:
                            self.fbx_files = gr.State([])

                # Right display area
                with gr.Column(scale=3):
                    self.output_display = gr.HTML(
                        value=get_placeholder_html(), show_label=False, elem_classes=["flask-display"]
                    )

            # Example Gallery Section
            with gr.Accordion("🎬 Example Gallery", open=True):
                self.example_gallery_display = gr.HTML(
                    value=get_example_gallery_grid_html(),
                    show_label=False,
                    elem_classes=["example-gallery-display"]
                )
                # Create use example buttons for each example
                with gr.Row():
                    self.use_example_btns = []
                    for idx, example in enumerate(EXAMPLE_GALLERY_LIST):
                        btn = gr.Button(
                            f"📋 Use Example {idx + 1}",
                            variant="secondary",
                            size="sm",
                        )
                        self.use_example_btns.append((btn, idx))

            # Footer
            gr.Markdown(FOOTER_MD, elem_classes=["footer"])

            # Hidden API endpoint for VSCode extension
            with gr.Row(visible=False):
                api_text = gr.Textbox()
                api_duration = gr.Number(value=5.0)
                api_seeds = gr.Textbox(value="0,1,2,3")
                api_cfg = gr.Number(value=5.0)
                api_output = gr.JSON()
                gr.Button().click(
                    fn=smpl_frames_api,
                    inputs=[api_text, api_duration, api_seeds, api_cfg],
                    outputs=[api_output],
                    api_name="smpl_frames",
                )

            self._bind_events()
            demo.load(fn=self._get_header_text, outputs=[self.header_md])
            return demo

    def _build_advanced_settings(self):
        with gr.Row():
            self.seed_input = gr.Textbox(
                label="🎯 Random Seeds",
                value="0,1,2,3",
                placeholder="e.g.: 0,1,2,3",
                scale=3,
            )
            self.dice_btn = gr.Button(
                "🎲",
                variant="secondary",
                size="sm",
                scale=1,
                min_width=50,
            )
        self.cfg_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5.0,
            step=0.1,
            label="⚙️ CFG Strength",
        )

    def _on_use_example(self, example_idx: int):
        """When clicking 'Use This Example' button, fill in the example prompt"""
        if example_idx < 0 or example_idx >= len(EXAMPLE_GALLERY_LIST):
            if self.prompt_engineering_available:
                return ("", "0,1,2,3", gr.update(), gr.update(value="", visible=False), gr.update(interactive=False), "Please select a valid example")
            else:
                return ("", "0,1,2,3", gr.update(), gr.update(), gr.update(), gr.update())

        example = EXAMPLE_GALLERY_LIST[example_idx]
        if self.prompt_engineering_available:
            # Set text directly to rewritten_text and enable generate button
            return (
                example["prompt"],
                example["seeds"],
                gr.update(value=example["duration"]),
                gr.update(value=example["prompt"], visible=True),
                gr.update(interactive=True),
                "✅ Example selected! Click [🚀 Generate Motion] to start.",
            )
        else:
            return (
                example["prompt"],
                example["seeds"],
                gr.update(value=example["duration"]),
                gr.update(),
                gr.update(),
                gr.update(),
            )

    def _bind_events(self):
        # Generate random seeds
        self.dice_btn.click(self._generate_random_seeds, outputs=[self.seed_input])

        # Use example buttons - bind each button to its example
        for btn, idx in self.use_example_btns:
            btn.click(
                fn=lambda i=idx: self._on_use_example(i),
                outputs=[self.text_input, self.seed_input, self.duration_slider, self.rewritten_text, self.generate_btn, self.status_output],
            )

        # Bind example selection event
        self.example_dropdown.change(
            fn=self._on_example_select,
            inputs=[self.example_dropdown],
            outputs=[self.text_input, self.seed_input, self.duration_slider, self.rewritten_text, self.generate_btn, self.status_output],
        )

        # Rewrite text logic (only bind when rewrite is available)
        if self.prompt_engineering_available:
            self.rewrite_btn.click(fn=lambda: "Rewriting text, please wait...", outputs=[self.status_output]).then(
                self._prompt_engineering,
                inputs=[
                    self.text_input,
                    self.duration_slider,
                ],
                outputs=[self.rewritten_text, self.generate_btn, self.duration_slider, self.status_output],
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[self.rewritten_text],
            )

        # Generate motion logic
        self.generate_btn.click(
            fn=lambda: "Generating motion, please wait... (It takes some extra time for the first generation)",
            outputs=[self.status_output],
        ).then(
            generate_motion_func,
            inputs=[self.text_input, self.rewritten_text, self.seed_input, self.duration_slider, self.cfg_slider],
            outputs=[self.output_display, self.fbx_files],
        ).then(
            fn=lambda fbx_list: (
                (
                    "🎉 Motion generation completed! You can view the motion visualization result on the right. FBX files are ready for download."
                    if fbx_list
                    else "🎉 Motion generation completed! You can view the motion visualization result on the right"
                ),
                gr.update(visible=bool(fbx_list)),
            ),
            inputs=[self.fbx_files],
            outputs=[self.status_output, self.fbx_download_row],
        )

        # Reset logic - different behavior based on rewrite availability
        if self.prompt_engineering_available:
            # When text_input changes:
            # - If text_input == rewritten_text, it means the change was triggered by example selection,
            #   so we should NOT hide the rewritten_text (keep it visible and generate button enabled)
            # - If text_input != rewritten_text, it means user manually edited the input,
            #   so we should hide the rewritten_text and require a new rewrite
            self.text_input.change(
                fn=lambda text, rewritten: (
                    gr.update() if text.strip() == rewritten.strip() else gr.update(visible=False),
                    gr.update() if text.strip() == rewritten.strip() else gr.update(interactive=False),
                    (
                        "✅ Example selected! Click [🚀 Generate Motion] to start."
                        if text.strip() == rewritten.strip() and text.strip()
                        else "Please click the [🔄 Rewrite Text] button to rewrite the text first"
                    ),
                ),
                inputs=[self.text_input, self.rewritten_text],
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        else:
            # When rewrite is not available, enable generate button directly when text is entered
            self.text_input.change(
                fn=lambda text: (
                    gr.update(visible=False),
                    gr.update(interactive=bool(text.strip())),
                    (
                        "Ready to generate! Click [🚀 Generate Motion] to start."
                        if text.strip()
                        else "Enter your text and click [🚀 Generate Motion] directly."
                    ),
                ),
                inputs=[self.text_input],
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        # Only bind rewritten_text change when rewrite is available
        if self.prompt_engineering_available:
            self.rewritten_text.change(
                fn=lambda text: (
                    gr.update(interactive=bool(text.strip())),
                    (
                        "Rewritten text has been modified, you can click [🚀 Generate Motion]"
                        if text.strip()
                        else "Rewritten text cannot be empty, please enter valid text"
                    ),
                ),
                inputs=[self.rewritten_text],
                outputs=[self.generate_btn, self.status_output],
            )


def create_demo(final_model_path):
    """Create the Gradio demo with Zero GPU support."""

    class Args:
        model_path = final_model_path
        output_dir = "output/gradio"
        use_prompt_engineering = USE_PROMPT_ENGINEERING
        use_text_encoder = True

    args = Args()

    # Check required files:
    cfg = osp.join(args.model_path, "config.yml")
    ckpt = osp.join(args.model_path, "latest.ckpt")
    if not osp.exists(cfg):
        raise FileNotFoundError(f">>> Configuration file not found: {cfg}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # For Zero GPU: Don't load model at startup, use lazy loading
    # Create a minimal runtime for UI initialization (without model loading)
    ui = T2MGradioUI(args=args)
    demo = ui.build_ui()
    return demo


# Create demo at module level for Hugging Face Spaces
# Pre-download text encoder models first (without loading)


if __name__ == "__main__":
    # Create demo at module level for Hugging Face Spaces
    import argparse
    parser = argparse.ArgumentParser(description="HY-Motion-1.0 Gradio App")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    USE_PROMPT_ENGINEERING = True
    try_to_download_text_encoder()
    # Then download the main model
    final_model_path = try_to_download_model()
    model_inference = ModelInference(final_model_path,
        use_prompt_engineering=False, use_text_encoder=True)
    model_inference.initialize_model(device="cpu")

    # Generate examples on first startup (if not exists)
    ensure_examples_generated(model_inference)

    demo = create_demo(final_model_path)
    demo.launch(server_name="0.0.0.0", server_port=args.port)
