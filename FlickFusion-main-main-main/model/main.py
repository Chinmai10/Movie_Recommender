from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class Model:
    def __init__(self):
        self.iter = 0
        self.movie = ['Tomb Raider','Big Bully','Fall Time','Batman'] # our movie list output for now
        self.cf_knn_model = NearestNeighbors(metric="cosine",algorithm="brute",n_neighbors=10,n_jobs=-1)
        
        self.movie_data = None
        self.movie_metadata = None
        
        self.user_ratings_df = None
        self.user_item_matrix = None
        self.tester = None
        self.answer = None
    def printAnswer(self):
        print(self.answer)

    def shortencsv(self): # make the csv files shorter #todo
        pass
    
    def make_csv(self):
        self.user_ratings_df = pd.read_csv('ratings.csv', low_memory=False)
        self.movie_data = pd.read_csv("movies_metadata.csv",low_memory=False)
        self.tester = pd.read_csv("ratings.csv",low_memory=False)
    def make_data(self):# manipulating the data

        self.movie_metadata = self.movie_data[["title","genres"]]
        self.movie_metadata = self.movie_metadata[:40000]

        self.movie_data = pd.concat([self.user_ratings_df,self.movie_metadata],axis=1).fillna(0)

        self.user_ratings_df = self.user_ratings_df.iloc[:40000]
        self.user_item_matrix = self.user_ratings_df.pivot(index=["userId"],columns=["movieId"],values='rating').fillna(0)

        self.movie_data = self.movie_data.dropna()
        self.movie_data = self.movie_data[:40000]

        
        #was creating duplicate columns soo removed the column indices ,combined and then given column indices.
        #some error with adding a empty row into the dataframe which is why this is done.
        cols= self.user_item_matrix.columns
        temp = self.user_item_matrix
        arr = temp.to_numpy()

        # print(arr.shape)

        tempa = np.zeros((1,5719))



        tempa = tempa.astype("int")
        arr=np.concatenate([arr,np.array(tempa)])
        i = arr.shape[0]
        self.user_item_matrix = arr
        t1 = pd.DataFrame(self.user_item_matrix,columns=cols)
        self.user_item_matrix = t1
        self.user_item_matrix = self.user_item_matrix.fillna(0)        

    def movie_recommender_engine(self,movie_name, matrix, cf_model, n_recs,rating=2):
        # Fit model on matrix
        self.cf_knn_model.fit(matrix)

        # Extract input movie ID

        movie_id = process.extractOne(movie_name, self.movie_data['title'])
        sl = int(movie_id[2])
        it = self.movie_data.iloc[sl]['movieId']

        # matrix[movie_id]

        self.user_item_matrix.iloc[len(self.user_item_matrix)-1][it]=rating
    
        # Calculate neighbour distances
        #returns us the 10 nearest nodes
        #indices are the movie ids
        #distances are the distances from the user node

        distances, indices = cf_model.kneighbors(np.array(self.user_item_matrix.iloc[len(self.user_item_matrix)-1]).reshape(1,-1), n_neighbors=n_recs)

        # print(indices)

        print(distances)
        movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])

        # print(movie_rec_ids)
        #sorts

        movie_rec_ids = movie_rec_ids[:0:-1]

        # print(movie_rec_ids)

        # # List to store recommendations

        cf_recs = []
        for i in movie_rec_ids:
            cf_recs.append({'Title':self.movie_data['title'][i[0]],'Distance':i[1],'Movie Id':i[0]})

        # selects top number of recommendations needed
        # no need of this last part u can just send it as a list to the website
        df = pd.DataFrame(cf_recs, index = range(1,n_recs))

        return df[:400]

    def model(self):
        self.cf_knn_model.fit(self.user_item_matrix)
        n_recs = 10
        ur_rating = float(input('enter the similarity rating(0-5):'))
        if ur_rating<=5 and ur_rating>=0:
            self.answer = self.movie_recommender_engine(self.movie[self.iter%len(self.movie)], self.user_item_matrix, self.cf_knn_model, n_recs,ur_rating)# this is where we send the input
            print(self.answer)
            print(self.user_item_matrix.iloc[len(self.user_item_matrix)-1].unique())
            self.iter+=1
            self.model()
        else:
            print("invalid rating")


d=Model()
d.make_csv()
d.make_data()
d.model()
# d.printAnswer()


