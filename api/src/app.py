from json import dumps as json_dumps
import traceback

from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse, Response

from .data_models import (
    validate_interactive_segment_single_slice_request_headers,
    validate_segment_single_slice_request_headers,
)
from .segmentor import Segmentor
from .utils import generate_task_id, np_arr_to_octet_stream, octet_stream_to_np_arr


segmentor = Segmentor("foobar", True)
app = FastAPI()


@app.post("/api/segment-single-slice")
async def segment_single_slice(request: Request):
    header_validation_errors = validate_segment_single_slice_request_headers(request.headers)
    if len(header_validation_errors) > 0:
        return JSONResponse(status_code=500, content={
            'detail': header_validation_errors,
        })
    
    volume_path = request.headers.get("X-IML-RELATIVE_VOLUME_PATH")
    slice_type = request.headers.get("X-IML-SLICE_TYPE")
    slice_index = int(request.headers.get("X-IML-SLICE_INDEX"))
    task_id = request.headers.get("X-IML-TASK_ID", generate_task_id())

    processing_success = False

    try:
        output_masks, index_to_label_map = segmentor.segment_single_slice(
            volume_path,
            slice_type,
            slice_index,
            task_id
        )

        response_stream = np_arr_to_octet_stream(output_masks)
        response_headers = {
            "X-IML-MASKS_INDEX_TO_LABEL_MAP": json_dumps(index_to_label_map),
        }

        processing_success = True
    except Exception:
        processing_success = False
        traceback.print_exc()
    
    if not processing_success:
        return JSONResponse(status_code=500, content={
            'success': False,
            'error': "Failed to segment slice."
        })

    return Response(
        content=response_stream,
        media_type="application/octet-stream",
        status_code=200,
        headers=response_headers
    )


@app.post("/api/interactive-segment-single-slice")
async def segment_single_slice(request: Request):
    header_validation_errors = validate_interactive_segment_single_slice_request_headers(request.headers)
    if len(header_validation_errors) > 0:
        return JSONResponse(status_code=500, content={
            'detail': header_validation_errors,
        })
    
    volume_path = request.headers.get("X-IML-RELATIVE_VOLUME_PATH")
    slice_type = request.headers.get("X-IML-SLICE_TYPE")
    slice_index = int(request.headers.get("X-IML-SLICE_INDEX"))
    task_id = request.headers.get("X-IML-TASK_ID", generate_task_id())

    processing_success = False

    try:
        request_body_binary_data = await request.body()
        
        segment_masks = octet_stream_to_np_arr(request_body_binary_data)
        
        output_masks, index_to_label_map = segmentor.interactive_segment_single_slice(
            volume_path,
            slice_type,
            slice_index,
            segment_masks,
            task_id
        )

        response_stream = np_arr_to_octet_stream(output_masks)
        response_headers = {
            "X-IML-MASKS_INDEX_TO_LABEL_MAP": json_dumps(index_to_label_map),
        }

        processing_success = True
    except Exception:
        processing_success = False
        traceback.print_exc()
    
    if not processing_success:
        return JSONResponse(status_code=500, content={
            'success': False,
            'error': "Failed to segment slice."
        })

    return Response(
        content=response_stream,
        media_type="application/octet-stream",
        status_code=200,
        headers=response_headers
    )
