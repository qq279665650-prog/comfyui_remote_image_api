# comfyui_remote_image_api

ComfyUI custom node for sending images to a remote image-generation API.

The main node is:

- Category: `Nkxx/API`
- Node: `Third Party Image POST API`

## Default Grsai Flow

The node now defaults to the same flow as the verified `grsai_nano_banana.py` script:

1. Upload input images to Grsai and collect image URLs.
2. POST a task to `https://grsai.dakka.com.cn/v1/draw/nano-banana`.
3. Poll `https://grsai.dakka.com.cn/v1/draw/result` with the returned task id.
4. Parse returned image URLs or base64 image data into ComfyUI `IMAGE`.

Recommended node values:

- `api_url`: `https://grsai.dakka.com.cn`
- `request_mode`: `grsai_nano_banana`
- `auth_mode`: `bearer`
- `model`: `nano-banana-2`
- `aspect_ratio`: `auto`, `16:9`, `9:16`, `4:3`, etc.
- `resolution`: `1K`, `2K`, or `4K`

The model alias `gemini-3.1-flash-image` is automatically mapped to Grsai model `nano-banana-2`.

## Inputs

- `api_url`: API base URL. For Grsai, use `https://grsai.dakka.com.cn`.
- `api_key`: Your API key. The key is only read from the node input and is not stored in the plugin.
- `model`: Model name. Default is `nano-banana-2`.
- `prompt`: Text prompt.
- `aspect_ratio`: Output ratio.
- `resolution`: Image size preset.
- `request_mode`: Use `grsai_nano_banana` for the verified Grsai flow.
- `auth_mode`: Use `bearer` for Grsai.
- `image_1` to `image_5`: Optional reference images.
- `extra_payload_json`: Optional JSON fields to merge into the Grsai task payload.
- `extra_headers_json`: Optional extra HTTP headers.
- `proxy_url`: Optional proxy URL. Leave empty for direct connection.
- `fallback_image`: Optional image returned when the request fails.

## Outputs

- `image`: Generated image.
- `status`: Success or error text.
- `failed`: `0` on success, `1` on failure.
- `video_conn`: Pass-through output kept for compatibility with existing workflows.
- `character_id`: Pass-through output kept for compatibility with existing workflows.

## Fallback Generic Modes

The older generic modes are still available:

- `json_base64`: Sends input images as base64 JSON.
- `multipart`: Sends input images as multipart form files.

For Grsai Nano Banana generation, use `grsai_nano_banana`.

## Install

Copy the whole `comfyui_remote_image_api` folder into ComfyUI's `custom_nodes` directory, then restart ComfyUI.

If dependencies are missing, run this in ComfyUI's Python environment:

```bash
pip install -r requirements.txt
```
