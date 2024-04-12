from flask import Flask, request, jsonify
from metaflow import FlowSpec  # Import FlowSpec and runner for Metaflow
from metaflow_workflow import RandomForestTraining, get_embedding  # Replace with actual import path
from storage import load_embeddings, store_embeddings
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

@app.route("/run", methods=["GET"])
def run_workflow():
    try:
        # Create a Metaflow flow instance with appropriate configuration (optional)
        flow = RandomForestTraining()
        # Uncomment and configure the following if needed:
        # with runner.LocalServer() as local_server:
        #     flow.run(local_server)
        flow.run()
        return jsonify({"message": "Workflow run successfully!"}), 200
    except Exception as e:
        return jsonify({"error": f"Error running workflow: {e}"}), 500

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Load embeddings from storage
        embeddings = load_embeddings()

        # Check if embeddings are loaded, handle potential absence gracefully
        if embeddings is None:
            return jsonify({"error": "Embeddings not found. Run /run first."}), 404

        # Access and use embeddings for prediction logic
        query = get_embedding("Machine Learning")  # Replace with user-provided query (if needed)
        embeddings["Similarity_Score"] = (embeddings["ada_embedding"].apply(lambda x: cosine_similarity([x], [query])[0][0]) * (3)) + (
            (embeddings["specialisation_embedding"].apply(lambda x: cosine_similarity([x], [query])[0][0])) * 4) + (
            (embeddings["primary_embedding"].apply(lambda x: cosine_similarity([x], [query])[0][0])) * (3)) + (
            (embeddings["Experience_Industry"].apply(lambda x: cosine_similarity([x], [query])[0][0])) * 2)
        result = embeddings["Similarity_Score"]
        return jsonify({"prediction": result}), 200  # Return top results or adjust as needed
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
