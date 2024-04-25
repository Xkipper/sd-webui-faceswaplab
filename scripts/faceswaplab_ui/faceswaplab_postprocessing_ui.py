from typing import List
import gradio as gr
from modules import shared
from scripts.faceswaplab_postprocessing.postprocessing_options import InpaintingWhen
from scripts.faceswaplab_utils.sd_utils import get_sd_option
from scripts.faceswaplab_ui.faceswaplab_inpainting_ui import face_inpainting_ui

import traceback
from scripts.faceswaplab_utils.typing import *
#import requests
from PIL import Image
import scripts.faceswaplab_swapping.swapper as swapper
from scripts.faceswaplab_utils.faceswaplab_logging import logger
import os
from modules import script_callbacks
from pathlib import Path
try:
    from send2trash import send2trash
    send2trash_installed = True
except ImportError:
    print("Delete Button: send2trash is not installed. recycle bin cannot be used.")
    send2trash_installed = False


def compareDos(img1: PILImage, img2: PILImage) -> str:
    """
    Compares the similarity between two faces extracted from images using cosine similarity.

    Args:
        img1: The first image containing a face.
        img2: The second image containing a face.

    Returns:
        A str of a float value representing the similarity between the two faces (0 to 1).
        Returns"You need 2 images to compare" if one or both of the images do not contain any faces.
    """
    try:
        if img1 is not None and img2 is not None:
            return str(round(float(swapper.compare_faces(img1, img2) * 100), 1)) + " %"
    except Exception as e:
        logger.error("Fail to compare", e)

        traceback.print_exc()
    return


def postprocessing_ui() -> List[gr.components.Component]:
    with gr.Tab(f"Global Post-Processing"):
        gr.Markdown(
            """Upscaling is performed on the whole image and all faces (including not swapped). Upscaling happens before face restoration."""
        )
        with gr.Row():
            face_restorer_name = gr.Radio(
                label="Restore Face",
                choices=["None"] + [x.name() for x in shared.face_restorers],
                value=get_sd_option(
                    "faceswaplab_pp_default_face_restorer",
                    shared.face_restorers[0].name(),
                ),
                type="value",
                elem_id="faceswaplab_pp_face_restorer",
            )

            with gr.Column():
                face_restorer_visibility = gr.Slider(
                    0,
                    1,
                    value=get_sd_option(
                        "faceswaplab_pp_default_face_restorer_visibility", 1
                    ),
                    step=0.001,
                    label="Restore visibility",
                    elem_id="faceswaplab_pp_face_restorer_visibility",
                )
                codeformer_weight = gr.Slider(
                    0,
                    1,
                    value=get_sd_option(
                        "faceswaplab_pp_default_face_restorer_weight", 1
                    ),
                    step=0.001,
                    label="codeformer weight",
                    elem_id="faceswaplab_pp_face_restorer_weight",
                )
        upscaler_name = gr.Dropdown(
            choices=[upscaler.name for upscaler in shared.sd_upscalers],
            value=get_sd_option("faceswaplab_pp_default_upscaler", "None"),
            label="Upscaler",
            elem_id="faceswaplab_pp_upscaler",
        )
        upscaler_scale = gr.Slider(
            1,
            8,
            1,
            step=0.1,
            label="Upscaler scale",
            elem_id="faceswaplab_pp_upscaler_scale",
        )
        upscaler_visibility = gr.Slider(
            0,
            1,
            value=get_sd_option("faceswaplab_pp_default_upscaler_visibility", 1),
            step=0.1,
            label="Upscaler visibility (if scale = 1)",
            elem_id="faceswaplab_pp_upscaler_visibility",
        )

        with gr.Accordion(label="Global-Inpainting (all faces)", open=False):
            gr.Markdown(
                "Inpainting sends image to inpainting with a mask on face (once for each faces)."
            )
            inpainting_when = gr.Dropdown(
                elem_id="faceswaplab_pp_inpainting_when",
                choices=[e.value for e in InpaintingWhen.__members__.values()],
                value=[InpaintingWhen.BEFORE_RESTORE_FACE.value],
                label="Enable/When",
            )
            global_inpainting = face_inpainting_ui("faceswaplab_gpp")
    # -------------------------------------------
    #   C O M P A R E   U I 
    # -------------------------------------------
    with gr.Tab("Compare"):        
        with gr.Row():
            attach_img1 = gr.Button("Attach generated to Face1", elem_id="faceswaplab_attach_btn")
            delete_btn = gr.Button("\U0000274c Delete generated IMG", elem_id="faceswaplab_delete_btn")
        
        delete_imagen_txt = gr.Markdown(""" _Skipper 2024_ """)

        gr.Markdown(""" ` """)

        with gr.Row():
            with gr.Blocks(show_footer=False) as blocks:
                imge1 = gr.components.Image(
				type="pil", label="Face 1", elem_id="faceswaplab_compare_face1",
			    )

            imge2 = gr.components.Image(
				type="pil", label="Face 2", elem_id="faceswaplab_compare_face2",
			)

        gr.Markdown(""" _ """)
        
        """ compare_result_text = gr.Textbox(
			interactive=False,
			label="Similarity",
			value="0",
			elem_id="faceswaplab_compare_result",
            container = True,
		) """
        compare_result_text = gr.Label(
            label="Similarity",
            elem_id="faceswaplab_compare_result",
            container = True,
            show_label = True,
        )
        attach_img1.click(attachGenerateIMG, outputs=[delete_imagen_txt, imge1])
        imge1.change(compareDos, inputs=[imge1, imge2], outputs=[compare_result_text])
        imge2.change(compareDos, inputs=[imge1, imge2], outputs=[compare_result_text])
        

        delete_btn.click(delete, outputs=[delete_imagen_txt])

    components = [
        face_restorer_name,
        face_restorer_visibility,
        codeformer_weight,
        upscaler_name,
        upscaler_scale,
        upscaler_visibility,
        inpainting_when,
    ] + global_inpainting

    # Ask sd to not store in ui-config.json
    for component in components:
        setattr(component, "do_not_save_to_config", True)
    return components


# ============================
#     DELETE IMG SECTION
# ============================
image_generate_path = ""


def attachGenerateIMG():
    global image_generate_path
    if os.path.exists(image_generate_path):
        image = Image.open(image_generate_path)
        return [str(image_generate_path), image]


def delete():
    global image_generate_path
    if os.path.exists(image_generate_path):
        if send2trash_installed:
            send2trash(image_generate_path)
            return str("Send to TRASH »» " + image_generate_path )
        else:
            file = Path(image_generate_path)
            file.unlink()
            return str("DELETED! »» " + image_generate_path )
    else:
        return str("~ERROR~ Nothing to Delete...")
    

def on_image_saved(params : script_callbacks.ImageSaveParams):
    global image_generate_path    
    image_generate_path = os.path.realpath(params.filename)
    

script_callbacks.on_image_saved(on_image_saved)


