import flask

app = flask.Flask(__name__)


@app.route("/ping", methods=["POST"])
def run_root() -> str:
    args = flask.request.get_json() or {}
    return {
        "response": "Your request was successful! 🎉",
        "args": args["message"],
    }


@app.route("/predict", methods=["POST"])
def run_predict() -> dict:
    import predict

    try:
        args = flask.request.get_json() or {}
        bucket = args["bucket"]
        model_dir = f"gs://{bucket}/fire-risk/model"
        data = args["data"]
        predictions = predict.run(data, model_dir)

        return {
            "method": "predict",
            "model_dir": model_dir,
            "predictions": predictions,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import os

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))