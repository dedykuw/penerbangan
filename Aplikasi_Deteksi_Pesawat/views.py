# views.py

import pandas as pd
from django.shortcuts import render
import numpy as np
import pickle


def upload_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file, delimiter=";")
        baroaltitude = df["baroaltitude"].to_numpy()
        X_unseen = []
        x = np.arange(0, len(baroaltitude))
        y = baroaltitude

        new_x = np.linspace(0, len(x), 127)
        X_unseen.append(np.interp(new_x, x, y.reshape(-1)))

        with open("model/Standart Scaler.pickle", "rb") as fsave:
            scaler = pickle.load(fsave)

        X_unseen_scaled = scaler.transform(X_unseen)

        with open("model/Model QDA.pickle", "rb") as fsave:
            model = pickle.load(fsave)

        y_unseen_predicted = model.predict(X_unseen_scaled)

        return render(request, 'result.html', {
            'df': df,
            'result' : 'Abnormal' if  y_unseen_predicted[0] == 1   else 'Normal'
            # 'result' : y_unseen_predicted[0]
        })

    return render(request, 'upload.html')
