from django.urls import path
from .views import query_ai

urlpatterns = [
    path('api/query/', query_ai, name='query_ai'),
]
