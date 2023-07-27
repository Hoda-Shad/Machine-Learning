class KNN:
    def __init__(self, k):
        self.k = k 

        
    def fit(self,x,y ): 
        self.X_train = x 
        self.Y_train = y 
        
    def predict(self,X): 
        for i in X :
            distances = self.euclidean_distance(i, X_train)
            nearest_neighbors = np.argsort(distances)[0:self.k]
        return np.argmax(np.bincount(self.Y_train(nearest_neighbors)))

    # def evaluate(x,y):
    #     X_test = x
    #     Y_test = y 
        
    #     return 

    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum( x1 - x2 )**2) 
        
        