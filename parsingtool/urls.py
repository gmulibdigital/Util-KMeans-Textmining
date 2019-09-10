from django.urls import path
from . import views

urlpatterns = [
    path('', views.simple_upload, name='simple_upload'),
    path('warning', views.showlog, name='warning'),
    # path('result', views.showresult, name='result'),
]