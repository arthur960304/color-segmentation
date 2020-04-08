import numpy as np

class LogisticRegression(object):
    def __init__(self, n_features):
        self.n_features = n_features
        
    def weightInit(self):
        w = np.zeros((1, self.n_features))
        b = 0
        return w,b

    def sigmoid(self, x):
        x = 1 / (1 + np.exp(-x))
        return x
        
    def optimize(self, w, b, X, Y):
        m = X.shape[0]

        # prediction
        final_result = self.sigmoid(np.dot(w,X.T)+b)
        cost = (-1/m)*(np.sum((Y.T*np.log(final_result)) + ((1-Y.T)*(np.log(1-final_result)))))

        # gradient calculation
        dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
        db = (1/m)*(np.sum(final_result-Y.T))

        grads = {"dw": dw, "db": db}

        return grads, cost
        
    
    def train(self, w, b, X, Y, lr, no_iterations):
        costs = []
        for _ in range(no_iterations):
            grads, cost = self.optimize(w,b,X,Y)
            dw = grads["dw"]
            db = grads["db"]
            
            # weight update
            w = w - (lr * (dw.T))
            b = b - (lr * db)

            costs.append(cost)
            #print("Cost after %i iteration is %f" %(i, cost))

        # final parameters
        coeff = {"w": w, "b": b}
        gradient = {"dw": dw, "db": db}

        return coeff, gradient, costs
        
        
    def predict(self, x, coeff):
        n = x.shape[0]
        w = coeff["w"]
        b = coeff["b"]
        y_pred = np.zeros((1,n))
        results = self.sigmoid(np.dot(w,x.T)+b)
        
        for i in range(n):
            if results[0][i] > 0.5:
                y_pred[0][i] = 1
                          
        return y_pred