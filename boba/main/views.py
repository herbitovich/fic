from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .forms import ImageUploadForm
import os
from .process import run
from .models import UploadedImage
from .comm import predict 
from ultralytics import YOLO
# Create your views here.
model = YOLO("main/yolov8_final.pt")
@csrf_exempt 
def home(request):
    if request.method == "GET":
        return render(request, 'home.html')
    elif request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            
            uploaded_image = UploadedImage.objects.create(image=image)
            image_path = uploaded_image.image.url[1:]
            save_image_url = image_path.replace('media', 'static').replace('uploads', 'img')
            bboxes = predict(image_path, model)
            print(image_path, save_image_url, bboxes)
            run(image_path, bboxes, os.path.dirname(save_image_url), save_image_url)
            return JsonResponse({'imageUrl': save_image_url})
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)