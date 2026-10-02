# ============================================
# Script: tinyunet_dashboard.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
#
# Purpose:
#   Interactive dashboard for the TinyUNet segmentation model
#   (trained by aiml_08_train_segmentation_demo.py, run by
#   aiml_09_run_trained_model_transparent.py).
#
#   Tabs:
#     1. Training Results  - load a results/*_history.json file and plot
#                             train/val loss and val IoU curves, plus a
#                             summary of hyperparameters and model size.
#     2. Run Inference      - pick a trained *_model.pt file, upload an
#                             image, and see the original image, the
#                             predicted mask, and the transparent-background
#                             composite side by side.
#
# Usage:
#   python tinyunet_dashboard.py
#   -> open http://127.0.0.1:8050
#
# Requirements:
#   pip install dash plotly torch opencv-python numpy pillow
# ============================================

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------
# Model definition - must match aiml_08_train_segmentation_demo.py /
# aiml_09_run_trained_model_transparent.py exactly, since we load
# weights trained with that architecture.
# ----------------------------------------------------
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)  # logits


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def list_history_files():
    return sorted(p.name for p in RESULTS_DIR.glob("*_history.json"))


def list_model_files():
    return sorted(p.name for p in RESULTS_DIR.glob("*_model.pt"))


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def decode_upload_to_bgr(contents: str) -> np.ndarray:
    """Decode a dcc.Upload 'contents' data URI into an OpenCV BGR array."""
    _, encoded = contents.split(",", 1)
    decoded = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(decoded)).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def predict_mask(model, img_bgr: np.ndarray, device: torch.device, threshold: float) -> np.ndarray:
    """Run TinyUNet on one image, return a full-resolution binary mask
    (0/255 uint8) matching the original image's dimensions."""
    orig_h, orig_w = img_bgr.shape[:2]

    resized = cv2.resize(img_bgr, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (probs > threshold).astype(np.uint8) * 255
    mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_full


# ----------------------------------------------------
# App
# ----------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "TinyUNet Dashboard"

TAB_STYLE = {"padding": "10px", "fontWeight": "600"}
SELECTED_TAB_STYLE = {"padding": "10px", "fontWeight": "600", "borderTop": "3px solid #2c7be5"}

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H2("TinyUNet Segmentation Dashboard"),
        html.P(
            "Project 2 - AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks. "
            "Inspect training runs and try the trained TinyUNet model on new images."
        ),
        dcc.Tabs(
            id="tabs",
            value="training",
            children=[
                dcc.Tab(label="Training Results", value="training", style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
                dcc.Tab(label="Run Inference", value="inference", style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
            ],
        ),
        html.Div(id="tab-content", style={"marginTop": "20px"}),
    ],
)


# ----------------------------------------------------
# Tab content
# ----------------------------------------------------
def training_tab():
    history_files = list_history_files()
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Training run (results/*_history.json):"),
                    dcc.Dropdown(
                        id="history-dropdown",
                        options=[{"label": f, "value": f} for f in history_files],
                        value=history_files[0] if history_files else None,
                        placeholder="No history files found in ./results",
                        style={"maxWidth": "500px"},
                    ),
                    html.Button("Refresh list", id="refresh-history-btn", n_clicks=0, style={"marginTop": "10px"}),
                ]
            ),
            html.Div(id="training-summary", style={"marginTop": "15px"}),
            dcc.Graph(id="loss-graph"),
            dcc.Graph(id="iou-graph"),
        ]
    )


def inference_tab():
    model_files = list_model_files()
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Trained model (results/*_model.pt):"),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[{"label": f, "value": f} for f in model_files],
                        value=model_files[0] if model_files else None,
                        placeholder="No model files found in ./results",
                        style={"maxWidth": "500px"},
                    ),
                    html.Button("Refresh list", id="refresh-model-btn", n_clicks=0, style={"marginTop": "10px", "marginRight": "10px"}),
                ]
            ),
            html.Div(
                [
                    html.Label("Foreground threshold:"),
                    dcc.Slider(id="threshold-slider", min=0.05, max=0.95, step=0.05, value=0.5,
                               marks={i / 10: str(i / 10) for i in range(1, 10)}),
                ],
                style={"maxWidth": "500px", "marginTop": "15px"},
            ),
            dcc.Upload(
                id="upload-image",
                children=html.Div(["Drag and drop an image, or ", html.A("select a file")]),
                style={
                    "width": "100%", "height": "70px", "lineHeight": "70px", "borderWidth": "1px",
                    "borderStyle": "dashed", "borderRadius": "6px", "textAlign": "center", "marginTop": "20px",
                },
                multiple=False,
            ),
            html.Div(id="inference-status", style={"marginTop": "10px", "color": "#555"}),
            html.Div(
                id="inference-output",
                style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap"},
            ),
        ]
    )


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "inference":
        return inference_tab()
    return training_tab()


# ----------------------------------------------------
# Training tab callbacks
# ----------------------------------------------------
@app.callback(
    Output("history-dropdown", "options"),
    Input("refresh-history-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_history_options(_n_clicks):
    return [{"label": f, "value": f} for f in list_history_files()]


@app.callback(
    Output("training-summary", "children"),
    Output("loss-graph", "figure"),
    Output("iou-graph", "figure"),
    Input("history-dropdown", "value"),
)
def update_training_view(history_filename):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white", height=350)

    if not history_filename:
        return html.Div("Select a training run to see its curves."), empty_fig, empty_fig

    history_path = RESULTS_DIR / history_filename
    if not history_path.is_file():
        return html.Div(f"File not found: {history_path}", style={"color": "red"}), empty_fig, empty_fig

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_iou = history.get("val_iou", [])

    summary = html.Div(
        [
            html.Span(f"Device: {history.get('device', 'n/a')}  |  "),
            html.Span(f"LR: {history.get('lr', 'n/a')}  |  "),
            html.Span(f"Batch size: {history.get('batch_size', 'n/a')}  |  "),
            html.Span(f"Params: {history.get('n_params', 'n/a'):,}  |  "
                      if isinstance(history.get("n_params"), int) else "Params: n/a  |  "),
            html.Span(f"Total time: {history.get('total_seconds', 0):.1f}s"),
            html.Br(),
            html.Span(
                f"Final: train_loss={train_loss[-1]:.4f}, val_loss={val_loss[-1]:.4f}, "
                f"val_iou={val_iou[-1]:.4f}" if train_loss and val_loss and val_iou else "No epoch data found."
            ),
        ]
    )

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train loss"))
    loss_fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val loss"))
    loss_fig.update_layout(title="Loss vs Epoch", xaxis_title="Epoch", yaxis_title="BCE loss",
                            template="plotly_white", height=350)

    iou_fig = go.Figure()
    iou_fig.add_trace(go.Scatter(x=epochs, y=val_iou, mode="lines+markers", name="Val IoU",
                                  line=dict(color="green")))
    iou_fig.update_layout(title="Validation IoU vs Epoch", xaxis_title="Epoch", yaxis_title="IoU",
                           yaxis_range=[0, 1], template="plotly_white", height=350)

    return summary, loss_fig, iou_fig


# ----------------------------------------------------
# Inference tab callbacks
# ----------------------------------------------------
@app.callback(
    Output("model-dropdown", "options"),
    Input("refresh-model-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_model_options(_n_clicks):
    return [{"label": f, "value": f} for f in list_model_files()]


@app.callback(
    Output("inference-output", "children"),
    Output("inference-status", "children"),
    Input("upload-image", "contents"),
    Input("model-dropdown", "value"),
    Input("threshold-slider", "value"),
    State("upload-image", "filename"),
    prevent_initial_call=True,
)
def run_inference(contents, model_filename, threshold, filename):
    triggered = callback_context.triggered_id if hasattr(callback_context, "triggered_id") else None

    if not contents:
        return [], "Upload an image to run inference."
    if not model_filename:
        return [], "Select a trained model (results/*_model.pt) first."

    model_path = RESULTS_DIR / model_filename
    if not model_path.is_file():
        return [], f"Model file not found: {model_path}"

    try:
        img_bgr = decode_upload_to_bgr(contents)
    except Exception as exc:
        return [], f"Could not read uploaded image: {exc}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = TinyUNet().to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()
    except Exception as exc:
        return [], f"Failed to load model '{model_filename}': {exc}"

    try:
        mask = predict_mask(model, img_bgr, device, threshold)
    except Exception as exc:
        return [], f"Inference failed: {exc}"

    # Build display images
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask])
    transparent_rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)

    original_uri = pil_to_data_uri(Image.fromarray(original_rgb))
    mask_uri = pil_to_data_uri(Image.fromarray(mask_rgb))
    transparent_uri = pil_to_data_uri(Image.fromarray(transparent_rgba, mode="RGBA"))

    def image_card(title, uri):
        return html.Div(
            [
                html.H4(title, style={"marginBottom": "6px"}),
                html.Img(src=uri, style={"maxWidth": "320px", "border": "1px solid #ddd", "borderRadius": "6px"}),
            ],
            style={"flex": "1 1 300px"},
        )

    cards = [
        image_card("Original", original_uri),
        image_card("Predicted mask", mask_uri),
        image_card("Transparent background", transparent_uri),
    ]

    status = f"Ran '{model_filename}' on '{filename}' (device={device}, threshold={threshold})."
    return cards, status


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
