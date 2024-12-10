import json
from typing import Any, Dict, List, Union

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..config import Config
from ..errors import CogError, ConfigDoesNotExist, PredictorNotSet
from ..schema import Status
from ..server.http import create_app
from ..suppress_output import suppress_output


def remove_title_next_to_ref(
    schema_node: Union[Dict[str, Any], List[Any]],
) -> Union[Dict[str, Any], List[Any]]:
    """
    Recursively remove 'title' from schema components that have a '$ref'.
    """
    if isinstance(schema_node, dict):
        if "$ref" in schema_node and "title" in schema_node:
            del schema_node["title"]
        for _key, value in schema_node.items():
            remove_title_next_to_ref(value)
    elif isinstance(schema_node, list):
        for i, item in enumerate(schema_node):
            schema_node[i] = remove_title_next_to_ref(item)
    return schema_node


# FastAPI 앱 생성
try:
    with suppress_output():
        app = create_app(cog_config=Config(), shutdown_event=None, is_build=True)
        if (
            app.state.setup_result
            and app.state.setup_result.status == Status.FAILED
        ):
            raise CogError(app.state.setup_result.logs)

        # OpenAPI 스키마 정리
        schema = remove_title_next_to_ref(app.openapi())

except ConfigDoesNotExist:
    raise ConfigDoesNotExist("no cog.yaml found or present") from None
except PredictorNotSet:
    raise PredictorNotSet("no predict method found in Predictor") from None


# FastAPI 앱에 엔드포인트 추가
@app.get("/openapi_schema", response_class=JSONResponse)
async def get_openapi_schema():
    """
    Endpoint to serve the OpenAPI schema.
    """
    try:
        return JSONResponse(content=schema, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
