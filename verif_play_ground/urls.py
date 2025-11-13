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

    #-------------------Chatbot Api--------------------------------------------------------------------
    path('api/chat', ChatAPIView.as_view(), name='chat'),

    #--------------------Cocotb simulation Api---------------------------------------------------------
    path("simulate/mux/download-excel", MuxSimulationExcelDownloadAPIView.as_view(), name="mux-sim-excel"),

    #--------------------Waveform Generation Api-------------------------------------------------------
    path("api/generate-waveform", WaveformGeneratorAPIView.as_view(), name="generate-waveform"),

    #-------------------Circuit Diagram Generation Api------------------------------
    path('api/circuit', MermaidCircuitAPIView.as_view(), name='generate_diagram'),

    #----------------------Logic explorer Api's --------------------------------------
    path("api/upload", UploadFileView.as_view(), name="upload"),
    path("api/explain", ExplainCodeView.as_view(), name="explain"),
    # path("api/testbench", GenerateTestbenchView.as_view(), name="testbench"),
    path("api/uvm", GenerateUVMView.as_view(), name="uvm"),
    path("api/report", DesignReportView.as_view(), name="report"),
    path('api/copy', CopyContentView.as_view(), name='copy_content'),
    path('api/clear', ClearAllView.as_view(), name='clear_all'),
    path('api/highlight', HighlightView.as_view(), name='highlight_code'),

    # Chunk-based endpoints (for large files or streaming editors)
    path('api/chunk/next', NextChunkView.as_view(), name='next_chunk'),
    path('api/chunk/add', AddChunkView.as_view(), name='add_chunk'),

    # Editor update & find/search
    path('api/update', UpdateCodeView.as_view(), name='update_code'),
    path('api/find', FindTextView.as_view(), name='find_text'),

]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
