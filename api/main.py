import os, tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from vision.model import load_model, parse_image

MAX_BYTES = 10 * 1024 * 1024
state = {}


def validate(p: dict) -> list[str]:
    n, errs, seen = p["n"], [], {}
    for i, cg in enumerate(p["cages"]):
        k = len(cg["cells"])
        label = f"cage {i} ({cg['target']}{cg['op']})"
        if cg["op"] == "?" or cg["target"] < 1:
            errs.append(f"{label}: unreadable clue, fix it in the JSON editor")
        if (cg["op"] == "=") != (k == 1):
            errs.append(f"{label}: operator does not fit {k} cell(s)")
        if cg["op"] in ("-", "/") and k != 2:
            errs.append(f"{label}: '{cg['op']}' needs exactly 2 cells, has {k}")
        if cg["op"] == "=" and not 1 <= cg["target"] <= n:
            errs.append(f"{label}: single value must be 1..{n}")
        for r, c in cg["cells"]:
            if not (0 <= r < n and 0 <= c < n):
                errs.append(f"{label}: cell {(r, c)} outside the board")
            elif (r, c) in seen:
                errs.append(f"cell {(r, c)} is in cages {seen[(r, c)]} and {i}")
            seen[(r, c)] = i
    missing = [(r, c) for r in range(n) for c in range(n) if (r, c) not in seen]
    if missing:
        errs.append(f"cells not covered by any cage: {missing[:6]}")
    return errs

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()          # loaded once, not per request
    yield


app = FastAPI(title="KenKen CV API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Cage(BaseModel):
    op: str
    target: int
    cells: list[list[int]]


class Puzzle(BaseModel):
    n: int
    cages: list[Cage]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")
    # parse_image takes a path, so write a temp file (cv2.imread detects format by content)
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        result = parse_image(tmp, state["model"])
    except Exception as e:
        raise HTTPException(422, f"Could not parse the board: {e}")
    finally:
        os.remove(tmp)
    result["unreadable"] = [i for i, c in enumerate(result["cages"]) if c["op"] == "?"]
    return result


@app.post("/solve")
def solve_endpoint(puzzle: Puzzle):
    data = puzzle.model_dump()
    try:
        errs = validate(data)
        if errs:
            raise HTTPException(422, "Board looks misread: " + " | ".join(errs[:5]))
        from solver.solver import solve
        cages = [(int(c["target"]), c["op"], [tuple(x) for x in c["cells"]])
                 for c in data["cages"]]
        grid = solve(data["n"], cages)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Solver error: {type(e).__name__}: {e}")
    return {"solution": [[int(v) for v in row] for row in grid]}