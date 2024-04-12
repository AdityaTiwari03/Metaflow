from metaflow import FlowSpec, step, resources
import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from storage import load_embeddings, store_embeddings

file = "./Book1.csv"


def get_embedding(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = str(text).replace("\n", " ")
    return model.encode(text, normalize_embeddings=False)


class RandomForestTraining(FlowSpec):
    data = pd.read_csv(file)

    @step
    def start(self):
        print("Started")
        self.X_Embeddings = self.data.apply(
            lambda row: ' '.join([str(row[col]) for col in self.data.columns if col not in ['TA_ID',
                                                                                        'Area_of_Specialisation',
                                                                                        'Primary_Skills',
                                                                                        'Patents',
                                                                                        'Publications']]), axis=1)
        self.next(self.embedding)

    @step
    def embedding(self):
        # Load embeddings from storage if available (new logic)
        self.data_embeddings = load_embeddings()
        if self.data_embeddings is None:
            # Create and store embeddings if not loaded
            data_embeddings = pd.DataFrame()
            data_embeddings["TA_ID"] = self.data["TA_ID"]
            data_embeddings['primary_embedding'] = self.data.Primary_Skills.apply(lambda x: get_embedding(x))
            data_embeddings['ada_embedding'] = self.X_Embeddings.apply(lambda x: get_embedding(x))
            data_embeddings['specialisation_embedding'] = self.data.Area_of_Specialisation.apply(lambda x: get_embedding(x))
            data_embeddings['Experience_Industry'] = self.data.Experience_Industry.apply(lambda x: get_embedding(x))
            self.data_embeddings = pd.DataFrame(data_embeddings)
            store_embeddings(self.data_embeddings)  # Store for future use

        self.next(self.query)

    @step
    def query(self):
        self.Query = get_embedding("Machine Learning")
        self.next(self.score)

    @step
    def score(self):
        embeddings = self.data_embeddings
        Query = self.Query
        embeddings["Similarity_Score"] = (embeddings["ada_embedding"].apply(lambda x: cosine_similarity([x], [Query])[0][0]) * (3)) + (
            (embeddings["specialisation_embedding"].apply(lambda x: cosine_similarity([x], [Query])[0][0])) * 4) + (
            (embeddings["primary_embedding"].apply(lambda x: cosine_similarity([x], [Query])[0][0])) * (3)) + (
            (embeddings["Experience_Industry"].apply(lambda x: cosine_similarity([x], [Query])[0][0])) * 2)
        Result = embeddings["Similarity_Score"]
        self.result = Result
        self.next(self.end)

    @step
    def end(self):
        print(self.result)
