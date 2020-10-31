#from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home,name='Home'),    
    path('model-deployment.html', views.modelDeployment,name='Model Deployment'),
    path('predict.html', views.prediction,name='Prediction'),
    path('dataset.html', views.dataset,name='Dataset'),
    path('future-scope.html', views.futureScope,name='Future Scope'),
    path('model-architecture.html', views.modelArchitecture,name='Model Architecture'),
    path('preprocessing.html', views.preprocessing,name='Prediction'),
    path('results.html', views.results,name='Results'),
    path('about-us.html', views.aboutUs,name='About Us')
]