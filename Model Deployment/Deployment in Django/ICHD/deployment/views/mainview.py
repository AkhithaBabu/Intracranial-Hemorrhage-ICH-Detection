from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .deploy import * 

#import deploy
# Create your views here.
def home(request):
    temp={'title':'Automated ICH Detection using Hybrid CNN model','bodyclass':' landing'}
    return render(request,'index.html',{'data':temp})

def modelDeployment(request):
    temp={'title':'Model Deployment','bodyclass':''}
    return render(request,'model-deployment.html',{'data':temp})

def prediction(request):    
    if request.method == 'POST' and request.FILES['CTSCAN']:
        folder='static/uploads' 
        myfile = request.FILES['CTSCAN']
        fs = FileSystemStorage(location=folder)
        fs.delete('CTSCAN.dcm')
        fs.delete('CTSCAN.png')
        #filename = fs.save(myfile.name, myfile)
        filename = fs.save('CTSCAN.dcm', myfile)
        if(myfile.name[-3:]!='dcm'):
            temp={'title':'Model Deployment','bodyclass':'','error':'Upload DCM file format only'}
            return render(request,'model-deployment.html',{'data':temp})
        uploaded_file_url = fs.url(filename)
        uploadedFile=folder+'/'+uploaded_file_url

        image_path = uploadedFile
        uploadpath=folder+'/CTSCAN.png'
        crop_head = CropHead()        
        model_path = 'static/resnet50_brain_00000033.h5'
        Image1 = dcm_to_png(image_path,crop =True,crop_head = crop_head)
        
        Image1 = Image.fromarray(Image1, 'RGB')
        Image1 = Image1.resize((224, 224), Image.NEAREST)
        Image1 = np.asarray(Image1)

        df=modelprediction(Image1,model_path)
        if(df['Any'].item()==0.0):
            message='Brain Hemorrhage Not Detected'
            columns=np.array(df.iloc[:,1:].columns)
            values=np.array(df.iloc[:,1:].values).ravel()
        else:
            message='Brain Hemorrhage Detected'
            columns=np.array(df.iloc[:,1:].columns)
            values=np.array(df.iloc[:,1:].values).ravel()
        
        visualize(Image1)
        


        temp={'title':'Prediction','bodyclass':'','CTSCAN':uploadpath,'show':df['Any'].item(),'msg':message,'columns':columns,'values':values}
        return render(request,'predict.html',{'data':temp})


#other pages
def dataset(request):
    temp={'title':'Dataset','bodyclass':''}
    return render(request,'dataset.html',{'data':temp})

def futureScope(request):
    temp={'title':'Future Scope','bodyclass':''}
    return render(request,'future-scope.html',{'data':temp})

def modelArchitecture(request):
    temp={'title':'Model Architecture','bodyclass':''}
    return render(request,'model-architecture.html',{'data':temp})

def preprocessing(request):
    temp={'title':'Pre-Processing','bodyclass':''}
    return render(request,'preprocessing.html',{'data':temp})

def results(request):
    temp={'title':'Result','bodyclass':''}
    return render(request,'results.html',{'data':temp})    

def aboutUs(request):
    temp={'title':'About Us','bodyclass':''}
    return render(request,'about-us.html',{'data':temp})
