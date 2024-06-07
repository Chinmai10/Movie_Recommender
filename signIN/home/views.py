from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from home.models import Indexsign
from django.contrib.auth.models import User



def index(request):
     return render(request,'index.html',)


def signUP(request):
     return render(request,'signUP.html',)





def indexsign(request):
     if request.method == "POST":
        name = request.POST.get("name")
        email= request.POST.get("email")
        password = request.POST.get("password")
         
        
        if name and email and password:
          new_indexsign = Indexsign(name=name, email=email , password=password)
          new_indexsign.save()
          messages.warning(request, "Your account as been successfully Signed Up!! ")
        
        if name and password:
          u =User.objects.get(username=name)
          
          # Authenticate the user
          User = authenticate(username=name, password=password)
          if User is not None:
                # If authentication is successful, log in the user and redirect to the home page
                login(request, User)
                messages.success(request, "You have been successfully logged in!")
                print("after login is successfull" , User)
                return redirect("/")
          else:
                # If authentication fails, display an error message
                print("after login is  not successfull" , User)
                
                messages.error(request, "Invalid username or password. Please try again.")
                return render(request, 'indexsign.html')
    # If the request method is not POST or authentication fails, render the sign-in page
     return render(request, 'indexsign.html')

     
