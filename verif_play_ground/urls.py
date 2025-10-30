"""
URL configuration for verif_play_ground project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from verif_play_ground_app.views import *

urlpatterns = [
    path('', home, name='home'),
    path('admin/', admin.site.urls),
    path('generate-uvm-ral/', UvmRalGeneratorView.as_view(), name='generate-uvm-ral'),
    path('generate-uvm-ral-base/', UvmRalGeneratorbase64View.as_view(), name='generate-uvm-ral-base'),
    path('drawSystemBlockAPIView/', DrawSystemBlockAPIView.as_view(), name='drawSystemBlockAPIView'),
    path('api/chat', ChatAPIView.as_view(), name='chat'),
    path("simulate/mux/download-excel", MuxSimulationExcelDownloadAPIView.as_view(), name="mux-sim-excel"),
    path("api/generate-waveform", WaveformGeneratorAPIView.as_view(), name="generate-waveform"),
    path('api/circuit', MermaidCircuitAPIView.as_view(), name='generate_diagram'),

]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
