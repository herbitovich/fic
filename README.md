
# boba - the Best Object Bounds Analysis

Сервис для обнаружения и классификации на полученном изображении опор ЛЭП. Приложение реализует возможность выделения объектов в ограничивающий прямоугольник, выдавая пользователю изображение с легендой классов и выделенными обнаруженными объектами.
Помимо этого сервис создаёт по выбранному пути директории с вырезанными объектами каждого класса.

# Способы использования
Мы реализовали два способа для комфортного использования нашего приложения:
* CLI
* Web-интерфейс

# Зависимости
Все зависимости указаны в файле requirements.txt  
Установка (из директории репозитория):  
```shell
pip install -r requirments.txt
```

# Про CLI

Запуск приложения происходит из директории ```boba/``` происходит следующим образом:
```shell
python -m main.process {АРГУМЕНТЫ} 
ОБЯЗАТЕЛЬНЫЕ АРГУМЕНТЫ:  
--image-path {путь_до/вашего/изображения}  
НЕОБЯЗАТЕЛЬНЫЕ АРГУМЕНТЫ:  
--output-dir {путь_до/директорий/с_классами}  
--save-image {путь_до/директории/куда/сохранится/обработанное/изображение}  
--show (флаг для показа обработанного изображения на экране)
```
Расширение обработанного изображения можно установить введя его вместе с названием  после "--save-image".  
Если ввести название сохраняемого изображения без разрешения, оно сохранится в формате {название}.png

Пример запуска:  
```shell
cd boba 
python -m main.process --image-path main/test.jpg --output-dir main/output --save-image main/test.png --show
```
В таком случае файлом для обработки будет выбран test.jpg из директории main, обработанное изображение test.png окажется там же. В директории main/output окажутся новые директории для каждого класса. В директории каждого из выявленных классов будут лежать вырезанные из исходного изображания кусочки с найденными объектами данного класса. Директория ```n```  создаётся только если на исходном изображении были выявлены ЛЭП класса ```n```. После обработки из-за присутствия флага `--show`, изображение появится в отдельном окне `matplotlib`. 

Для получения дополнительной информации о флагах, можно прописать:
```shell
python -m main.process -h
```
# Про web-интерфейс
Мы реализовали интуитивно понятный и удобный пользователю интерфейс:  
* Загружаем нужное изображение
* Нажимаем на стрелочку и ждём обработки
* Получаем обработанное изображение на экране

Для его запуска, необходимо прописать в терминале:
```shell
python boba/manage.py runserver
```
Адрес локального сервера появится в коммандной строке, вставьте его в ваш браузер и начинайте использовать.
# Архитектура репозитория
Директория `bob` -- папка, объединяющая в себе исходные файлы обоих типов интерфейсов

Файлы `initialTrain.py`, `YOLOv8wEmb.py` -- исходные файлы тренировки модели на этапе файн-тюна на датасете и этапе файн-тюна на датасете+metric-learning loss соответственно. 

Файлы `yolov8n.pt`, `yolov8_trained.pt`, `boba/main/yolov8_final.pt` -- исходные файлы модели до тренировки, после первого и второго этапов соответственно.

Остальные файлы репозитория -- вспомогательные скрипты, включающие в себя скрипт аугментации данных, скрипт составления датасета, `dataset.yaml`.

#Датасет
Можно найти по ссылке `https://drive.google.com/drive/folders/1Z2s9AZwCB5_lpNqKSa0gJaqVAp1z8Zld`, с папками `images` и `labels` для изображений и их аннотаций в формате YOLO.
# Авторы 

- [@herbitovich](https://www.github.com/herbitovich) Матвей
- [@ProteusP](https://github.com/ProteusP) Дмитрий
