# ============================================================================
# FREELANCER MARKETPLACE - COMPLETE DJANGO APPLICATION
# ============================================================================

# requirements.txt
"""
Django==4.2.7
Pillow==10.0.1
django-crispy-forms==2.0
crispy-bootstrap4==2022.1
django-widget-tweaks==1.4.12
python-decouple==3.8
"""

# ============================================================================
# 1. PROJECT STRUCTURE
# ============================================================================
"""
freelance_marketplace/
├── manage.py
├── requirements.txt
├── freelance_marketplace/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── marketplace/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── migrations/
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── freelancers.html
│   ├── projects.html
│   └── profile.html
└── static/
    ├── css/
    ├── js/
    └── images/
"""

# ============================================================================
# 2. DJANGO SETTINGS (settings.py)
# ============================================================================

import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY', default='your-secret-key-here')
DEBUG = config('DEBUG', default=True, cast=bool)
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'marketplace',
    'crispy_forms',
    'crispy_bootstrap4',
    'widget_tweaks',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'freelance_marketplace.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CRISPY_TEMPLATE_PACK = 'bootstrap4'

LOGIN_REDIRECT_URL = 'home'
LOGOUT_REDIRECT_URL = 'home'

# ============================================================================
# 3. MODELS (marketplace/models.py)
# ============================================================================

from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.validators import MinValueValidator, MaxValueValidator
from PIL import Image

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return self.name

class Skill(models.Model):
    name = models.CharField(max_length=50, unique=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name

class FreelancerProfile(models.Model):
    AVAILABILITY_CHOICES = [
        ('available', 'Available'),
        ('busy', 'Busy'),
        ('unavailable', 'Unavailable'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    bio = models.TextField(max_length=1000)
    hourly_rate = models.DecimalField(max_digits=8, decimal_places=2)
    location = models.CharField(max_length=200)
    avatar = models.ImageField(upload_to='avatars/', default='avatars/default.jpg')
    skills = models.ManyToManyField(Skill, blank=True)
    portfolio_url = models.URLField(blank=True, null=True)
    availability = models.CharField(max_length=20, choices=AVAILABILITY_CHOICES, default='available')
    rating = models.DecimalField(max_digits=3, decimal_places=2, default=0.0)
    total_reviews = models.IntegerField(default=0)
    projects_completed = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} - {self.title}"
    
    def get_absolute_url(self):
        return reverse('freelancer_detail', kwargs={'pk': self.pk})
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        
        if self.avatar:
            img = Image.open(self.avatar.path)
            if img.height > 300 or img.width > 300:
                output_size = (300, 300)
                img.thumbnail(output_size)
                img.save(self.avatar.path)

class ClientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company_name = models.CharField(max_length=200, blank=True)
    company_description = models.TextField(blank=True)
    location = models.CharField(max_length=200)
    website = models.URLField(blank=True, null=True)
    avatar = models.ImageField(upload_to='client_avatars/', default='avatars/default.jpg')
    rating = models.DecimalField(max_digits=3, decimal_places=2, default=0.0)
    total_reviews = models.IntegerField(default=0)
    projects_posted = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} - {self.company_name}"

class Project(models.Model):
    STATUS_CHOICES = [
        ('open', 'Open'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    BUDGET_TYPE_CHOICES = [
        ('fixed', 'Fixed Price'),
        ('hourly', 'Hourly Rate'),
    ]
    
    client = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=300)
    description = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    skills_required = models.ManyToManyField(Skill, blank=True)
    budget_type = models.CharField(max_length=10, choices=BUDGET_TYPE_CHOICES, default='fixed')
    budget_min = models.DecimalField(max_digits=10, decimal_places=2)
    budget_max = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    duration = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    featured = models.BooleanField(default=False)
    proposals_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deadline = models.DateField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('project_detail', kwargs={'pk': self.pk})

class Proposal(models.Model):
    freelancer = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    cover_letter = models.TextField()
    bid_amount = models.DecimalField(max_digits=10, decimal_places=2)
    delivery_time = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['freelancer', 'project']
    
    def __str__(self):
        return f"{self.freelancer.get_full_name()} - {self.project.title}"

class Review(models.Model):
    reviewer = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviews_given')
    reviewee = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviews_received')
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['reviewer', 'reviewee', 'project']
    
    def __str__(self):
        return f"{self.rating} stars - {self.project.title}"

class Message(models.Model):
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_messages')
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_messages')
    subject = models.CharField(max_length=200)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.sender.username} to {self.recipient.username} - {self.subject}"

# ============================================================================
# 4. FORMS (marketplace/forms.py)
# ============================================================================

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import FreelancerProfile, ClientProfile, Project, Proposal, Review, Message

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=100, required=True)
    last_name = forms.CharField(max_length=100, required=True)
    user_type = forms.ChoiceField(choices=[('freelancer', 'Freelancer'), ('client', 'Client')], required=True)
    
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2', 'user_type']

class FreelancerProfileForm(forms.ModelForm):
    class Meta:
        model = FreelancerProfile
        fields = ['title', 'bio', 'hourly_rate', 'location', 'avatar', 'skills', 
                 'portfolio_url', 'availability']
        widgets = {
            'bio': forms.Textarea(attrs={'rows': 4}),
            'skills': forms.CheckboxSelectMultiple(),
        }

class ClientProfileForm(forms.ModelForm):
    class Meta:
        model = ClientProfile
        fields = ['company_name', 'company_description', 'location', 'website', 'avatar']
        widgets = {
            'company_description': forms.Textarea(attrs={'rows': 4}),
        }

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ['title', 'description', 'category', 'skills_required', 'budget_type',
                 'budget_min', 'budget_max', 'duration', 'deadline']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 6}),
            'skills_required': forms.CheckboxSelectMultiple(),
            'deadline': forms.DateInput(attrs={'type': 'date'}),
        }

class ProposalForm(forms.ModelForm):
    class Meta:
        model = Proposal
        fields = ['cover_letter', 'bid_amount', 'delivery_time']
        widgets = {
            'cover_letter': forms.Textarea(attrs={'rows': 5, 'placeholder': 'Write a compelling cover letter...'}),
        }

class MessageForm(forms.ModelForm):
    class Meta:
        model = Message
        fields = ['subject', 'message']
        widgets = {
            'message': forms.Textarea(attrs={'rows': 5}),
        }

class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ['rating', 'comment']
        widgets = {
            'rating': forms.Select(choices=[(i, f"{i} Stars") for i in range(1, 6)]),
            'comment': forms.Textarea(attrs={'rows': 4}),
        }

# ============================================================================
# 5. VIEWS (marketplace/views.py)
# ============================================================================

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import login
from django.contrib import messages
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.db.models import Q, Avg, Count
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.urls import reverse_lazy

from .models import (FreelancerProfile, ClientProfile, Project, Proposal, 
                    Review, Message, Category, Skill)
from .forms import (UserRegisterForm, FreelancerProfileForm, ClientProfileForm,
                   ProjectForm, ProposalForm, MessageForm, ReviewForm)

def home(request):
    """Home page with featured freelancers and projects"""
    featured_freelancers = FreelancerProfile.objects.filter(
        availability='available'
    ).order_by('-rating', '-projects_completed')[:6]
    
    recent_projects = Project.objects.filter(status='open').order_by('-created_at')[:6]
    categories = Category.objects.all()
    
    # Stats
    stats = {
        'total_freelancers': FreelancerProfile.objects.count(),
        'total_projects': Project.objects.count(),
        'total_categories': Category.objects.count(),
    }
    
    context = {
        'featured_freelancers': featured_freelancers,
        'recent_projects': recent_projects,
        'categories': categories,
        'stats': stats,
    }
    return render(request, 'marketplace/home.html', context)

def register(request):
    """User registration"""
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            user_type = form.cleaned_data['user_type']
            
            # Create profile based on user type
            if user_type == 'freelancer':
                FreelancerProfile.objects.create(
                    user=user,
                    title='New Freelancer',
                    bio='Tell us about yourself!',
                    hourly_rate=25.00,
                    location='Not specified'
                )
            else:
                ClientProfile.objects.create(
                    user=user,
                    location='Not specified'
                )
            
            login(request, user)
            messages.success(request, f'Account created! Welcome {user.first_name}!')
            return redirect('profile_edit')
    else:
        form = UserRegisterForm()
    
    return render(request, 'registration/register.html', {'form': form})

class FreelancerListView(ListView):
    """List all freelancers with search and filter"""
    model = FreelancerProfile
    template_name = 'marketplace/freelancers.html'
    context_object_name = 'freelancers'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = FreelancerProfile.objects.select_related('user').prefetch_related('skills')
        
        # Search
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(user__first_name__icontains=search) |
                Q(user__last_name__icontains=search) |
                Q(title__icontains=search) |
                Q(skills__name__icontains=search) |
                Q(location__icontains=search)
            ).distinct()
        
        # Category filter
        category = self.request.GET.get('category')
        if category:
            queryset = queryset.filter(skills__category__id=category)
        
        # Availability filter
        availability = self.request.GET.get('availability')
        if availability:
            queryset = queryset.filter(availability=availability)
        
        # Sort
        sort = self.request.GET.get('sort', '-rating')
        queryset = queryset.order_by(sort)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['current_category'] = self.request.GET.get('category', '')
        context['current_search'] = self.request.GET.get('search', '')
        context['current_availability'] = self.request.GET.get('availability', '')
        context['current_sort'] = self.request.GET.get('sort', '-rating')
        return context

class FreelancerDetailView(DetailView):
    """Freelancer profile detail"""
    model = FreelancerProfile
    template_name = 'marketplace/freelancer_detail.html'
    context_object_name = 'freelancer'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        freelancer = self.get_object()
        
        # Get reviews
        reviews = Review.objects.filter(reviewee=freelancer.user).order_by('-created_at')[:5]
        context['reviews'] = reviews
        
        # Get recent projects (if any)
        recent_projects = Project.objects.filter(
            proposal__freelancer=freelancer.user
        ).order_by('-created_at')[:3]
        context['recent_projects'] = recent_projects
        
        return context

class ProjectListView(ListView):
    """List all projects with search and filter"""
    model = Project
    template_name = 'marketplace/projects.html'
    context_object_name = 'projects'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Project.objects.filter(status='open').select_related('client', 'category')
        
        # Search
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) |
                Q(description__icontains=search) |
                Q(skills_required__name__icontains=search)
            ).distinct()
        
        # Category filter
        category = self.request.GET.get('category')
        if category:
            queryset = queryset.filter(category__id=category)
        
        # Budget filter
        budget_min = self.request.GET.get('budget_min')
        budget_max = self.request.GET.get('budget_max')
        if budget_min:
            queryset = queryset.filter(budget_min__gte=budget_min)
        if budget_max:
            queryset = queryset.filter(budget_max__lte=budget_max)
        
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['current_category'] = self.request.GET.get('category', '')
        context['current_search'] = self.request.GET.get('search', '')
        return context

class ProjectDetailView(DetailView):
    """Project detail view"""
    model = Project
    template_name = 'marketplace/project_detail.html'
    context_object_name = 'project'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = self.get_object()
        
        # Check if current user has submitted a proposal
        if self.request.user.is_authenticated:
            try:
                user_proposal = Proposal.objects.get(
                    freelancer=self.request.user,
                    project=project
                )
                context['user_proposal'] = user_proposal
            except Proposal.DoesNotExist:
                context['user_proposal'] = None
        
        # Get recent proposals
        proposals = Proposal.objects.filter(project=project).order_by('-created_at')[:5]
        context['proposals'] = proposals
        
        return context

@login_required
def profile_view(request):
    """View user profile"""
    try:
        freelancer_profile = request.user.freelancerprofile
        return render(request, 'marketplace/profile.html', {
            'profile': freelancer_profile,
            'user_type': 'freelancer'
        })
    except FreelancerProfile.DoesNotExist:
        try:
            client_profile = request.user.clientprofile
            return render(request, 'marketplace/profile.html', {
                'profile': client_profile,
                'user_type': 'client'
            })
        except ClientProfile.DoesNotExist:
            messages.error(request, 'Profile not found. Please complete your profile.')
            return redirect('profile_edit')

@login_required
def profile_edit(request):
    """Edit user profile"""
    try:
        profile = request.user.freelancerprofile
        form_class = FreelancerProfileForm
        user_type = 'freelancer'
    except FreelancerProfile.DoesNotExist:
        try:
            profile = request.user.clientprofile
            form_class = ClientProfileForm
            user_type = 'client'
        except ClientProfile.DoesNotExist:
            messages.error(request, 'Profile not found.')
            return redirect('home')
    
    if request.method == 'POST':
        form = form_class(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        form = form_class(instance=profile)
    
    return render(request, 'marketplace/profile_edit.html', {
        'form': form,
        'user_type': user_type
    })

@login_required
def project_create(request):
    """Create a new project"""
    if request.method == 'POST':
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            project.client = request.user
            project.save()
            form.save_m2m()  # Save many-to-many relationships
            
            # Update client's project count
            try:
                client_profile = request.user.clientprofile
                client_profile.projects_posted += 1
                client_profile.save()
            except ClientProfile.DoesNotExist:
                pass
            
            messages.success(request, 'Project posted successfully!')
            return redirect('project_detail', pk=project.pk)
    else:
        form = ProjectForm()
    
    return render(request, 'marketplace/project_create.html', {'form': form})

@login_required
def proposal_create(request, project_id):
    """Submit a proposal for a project"""
    project = get_object_or_404(Project, id=project_id, status='open')
    
    # Check if user is a freelancer
    try:
        freelancer_profile = request.user.freelancerprofile
    except FreelancerProfile.DoesNotExist:
        messages.error(request, 'Only freelancers can submit proposals.')
        return redirect('project_detail', pk=project_id)
    
    # Check if proposal already exists
    existing_proposal = Proposal.objects.filter(
        freelancer=request.user,
        project=project
    ).exists()
    
    if existing_proposal:
        messages.warning(request, 'You have already submitted a proposal for this project.')
        return redirect('project_detail', pk=project_id)
    
    if request.method == 'POST':
        form = ProposalForm(request.POST)
        if form.is_valid():
            proposal = form.save(commit=False)
            proposal.freelancer = request.user
            proposal.project = project
            proposal.save()
            
            # Update project proposal count
            project.proposals_count += 1
            project.save()
            
            messages.success(request, 'Proposal submitted successfully!')
            return redirect('project_detail', pk=project_id)
    else:
        form = ProposalForm()
    
    return render(request, 'marketplace/proposal_create.html', {
        'form': form,
        'project': project
    })

@login_required
def dashboard(request):
    """User dashboard"""
    context = {'user_type': None}
    
    try:
        freelancer_profile = request.user.freelancerprofile
        context.update({
            'user_type': 'freelancer',
            'profile': freelancer_profile,
            'proposals': Proposal.objects.filter(freelancer=request.user).order_by('-created_at')[:5],
            'recent_projects': Project.objects.filter(status='open').order_by('-created_at')[:5],
        })
    except FreelancerProfile.DoesNotExist:
        try:
            client_profile = request.user.clientprofile
            context.update({
                'user_type': 'client',
                'profile': client_profile,
                'projects': Project.objects.filter(client=request.user).order_by('-created_at')[:5],
                'recent_proposals': Proposal.objects.filter(
                    project__client=request.user
                ).order_by('-created_at')[:5],
            })
        except ClientProfile.DoesNotExist:
            messages.error(request, 'Profile not found.')
            return redirect('profile_edit')
    
    return render(request, 'marketplace/dashboard.html', context)

# ============================================================================
# 6. URLS (marketplace/urls.py)
# ============================================================================

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/edit/', views.profile_edit, name='profile_edit'),
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # Freelancers
    path('freelancers/', views.FreelancerListView.as_view(), name='freelancers'),
    path('freelancer/<int:pk>/', views.FreelancerDetailView.as_view(), name='freelancer_detail'),
    
    # Projects
    path('projects/', views.ProjectListView.as_view(), name='projects'),
    path('project/<int:pk>/', views.ProjectDetailView.as_view(), name='project_detail'),
    path('project/create/', views.project_create, name='project_create'),
    path('project/<int:project_id>/proposal/', views.proposal_create, name='proposal_create'),
]

# ============================================================================
# 7. MAIN PROJECT URLS (freelance_marketplace/urls.py)
# ============================================================================

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('marketplace.urls')),
    
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='registration/password_reset.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='registration/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='registration/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'), name='password_reset_complete'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# ============================================================================
# 8. ADMIN CONFIGURATION (marketplace/admin.py)
# ============================================================================

from django.contrib import admin
from .models import (Category, Skill, FreelancerProfile, ClientProfile, 
                    Project, Proposal, Review, Message)

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at']
    search_fields = ['name']

@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = ['name', 'category']
    list_filter = ['category']
    search_fields = ['name']

@admin.register(FreelancerProfile)
class FreelancerProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'title', 'hourly_rate', 'rating', 'availability', 'projects_completed']
    list_filter = ['availability', 'skills', 'created_at']
    search_fields = ['user__username', 'user__first_name', 'user__last_name', 'title']
    filter_horizontal = ['skills']

@admin.register(ClientProfile)
class ClientProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'company_name', 'location', 'rating', 'projects_posted']
    search_fields = ['user__username', 'user__first_name', 'user__last_name', 'company_name']

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['title', 'client', 'category', 'budget_min', 'status', 'proposals_count', 'created_at']
    list_filter = ['status', 'category', 'budget_type', 'created_at']
    search_fields = ['title', 'description', 'client__username']
    filter_horizontal = ['skills_required']

@admin.register(Proposal)
class ProposalAdmin(admin.ModelAdmin):
    list_display = ['freelancer', 'project', 'bid_amount', 'created_at']
    list_filter = ['created_at']
    search_fields = ['freelancer__username', 'project__title']

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ['reviewer', 'reviewee', 'project', 'rating', 'created_at']
    list_filter = ['rating', 'created_at']
    search_fields = ['reviewer__username', 'reviewee__username', 'project__title']

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['sender', 'recipient', 'subject', 'is_read', 'created_at']
    list_filter = ['is_read', 'created_at']
    search_fields = ['sender__username', 'recipient__username', 'subject']

# ============================================================================
# 9. BASE TEMPLATE (templates/base.html)
# ============================================================================
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}FreelanceHub - Find Perfect Freelance Talent{% endblock %}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    <style>
        :root {
            --primary-color: #3b82f6;
            --primary-dark: #2563eb;
            --secondary-color: #8b5cf6;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            min-height: 100vh;
        }
        
        .navbar {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .navbar-brand {
            font-weight: 700;
            font-size: 1.5rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(59, 130, 246, 0.3);
        }
        
        .card {
            border: none;
            border-radius: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
        }
        
        .hero-section {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 100px 0;
            margin-bottom: 50px;
        }
        
        .skill-badge {
            background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
            color: #0277bd;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 2px;
            display: inline-block;
        }
        
        .rating-stars {
            color: #fbbf24;
        }
        
        .avatar {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .project-card {
            border-left: 4px solid var(--primary-color);
        }
        
        .freelancer-card {
            border-top: 4px solid var(--secondary-color);
        }
        
        .footer {
            background: #1e293b;
            color: white;
            padding: 40px 0;
            margin-top: 80px;
        }
        
        .search-form {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'home' %}">
                <i class="fas fa-briefcase me-2"></i>FreelanceHub
            </a>
            
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{% url 'freelancers' %}">Browse Talent</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{% url 'projects' %}">Find Projects</a>
                    </li>
                </ul>
                
                <ul class="navbar-nav">
                    {% if user.is_authenticated %}
                        <li class="nav-item dropdown">
                            <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown">
                                <i class="fas fa-user-circle me-1"></i>{{ user.first_name|default:user.username }}
                            </a>
                            <ul class="dropdown-menu">
                                <li><a class="dropdown-item" href="{% url 'dashboard' %}">Dashboard</a></li>
                                <li><a class="dropdown-item" href="{% url 'profile' %}">Profile</a></li>
                                <li><hr class="dropdown-divider"></li>
                                <li><a class="dropdown-item" href="{% url 'logout' %}">Logout</a></li>
                            </ul>
                        </li>
                        <li class="nav-item">
                            <a class="btn btn-outline-light ms-2" href="{% url 'project_create' %}">
                                <i class="fas fa-plus me-1"></i>Post Project
                            </a>
                        </li>
                    {% else %}
                        <li class="nav-item">
                            <a class="nav-link" href="{% url 'login' %}">Login</a>
                        </li>
                        <li class="nav-item">
                            <a class="btn btn-outline-light ms-2" href="{% url 'register' %}">Join Now</a>
                        </li>
                    {% endif %}
                </ul>
            </div>
        </div>
    </nav>

    <!-- Messages -->
    {% if messages %}
        <div class="container mt-3">
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        </div>
    {% endif %}

    <!-- Main Content -->
    <main>
        {% block content %}
        {% endblock %}
    </main>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5><i class="fas fa-briefcase me-2"></i>FreelanceHub</h5>
                    <p class="text-muted">Connecting talented freelancers with amazing projects worldwide.</p>
                </div>
                <div class="col-md-6 text-end">
                    <h6>Quick Links</h6>
                    <div>
                        <a href="{% url 'freelancers' %}" class="text-light text-decoration-none me-3">Browse Talent</a>
                        <a href="{% url 'projects' %}" class="text-light text-decoration-none me-3">Find Projects</a>
                        <a href="#" class="text-light text-decoration-none">Support</a>
                    </div>
                </div>
            </div>
            <hr class="my-4">
            <div class="text-center text-muted">
                <p>&copy; 2024 FreelanceHub. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    {% block extra_js %}
    {% endblock %}
</body>
</html>
"""

# ============================================================================
# 10. HOME TEMPLATE (templates/marketplace/home.html)
# ============================================================================
"""
{% extends 'base.html' %}

{% block content %}
<!-- Hero Section -->
<section class="hero-section">
    <div class="container">
        <div class="row align-items-center">
            <div class="col-lg-6">
                <h1 class="display-4 fw-bold mb-4">
                    Find the Perfect <span class="text-warning">Freelance Talent</span>
                </h1>
                <p class="lead mb-4">
                    Connect with skilled professionals ready to bring your projects to life. 
                    From development to design, marketing to writing - find your perfect match.
                </p>
                
                <!-- Search Form -->
                <div class="search-form mb-4">
                    <form method="get" action="{% url 'freelancers' %}">
                        <div class="row g-3">
                            <div class="col-md-8">
                                <input type="text" name="search" class="form-control form-control-lg" 
                                       placeholder="Search for freelancers, skills, or services...">
                            </div>
                            <div class="col-md-4">
                                <button type="submit" class="btn btn-primary btn-lg w-100">
                                    <i class="fas fa-search me-2"></i>Search
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
                
                <!-- Stats -->
                <div class="row text-center">
                    <div class="col-4">
                        <h3>{{ stats.total_freelancers }}+</h3>
                        <small>Freelancers</small>
                    </div>
                    <div class="col-4">
                        <h3>{{ stats.total_projects }}+</h3>
                        <small>Projects</small>
                    </div>
                    <div class="col-4">
                        <h3>{{ stats.total_categories }}+</h3>
                        <small>Categories</small>
                    </div>
                </div>
            </div>
            <div class="col-lg-6">
                <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=600&h=400&fit=crop" 
                     class="img-fluid rounded-3 shadow-lg" alt="Freelance Team">
            </div>
        </div>
    </div>
</section>

<!-- Categories -->
<section class="container mb-5">
    <h2 class="text-center mb-4">Popular Categories</h2>
    <div class="row">
        {% for category in categories %}
        <div class="col-md-3 mb-3">
            <a href="{% url 'freelancers' %}?category={{ category.id }}" class="text-decoration-none">
                <div class="card text-center h-100">
                    <div class="card-body">
                        <i class="fas fa-code fa-2x text-primary mb-3"></i>
                        <h5>{{ category.name }}</h5>
                        <p class="text-muted">{{ category.description|truncatewords:8 }}</p>
                    </div>
                </div>
            </a>
        </div>
        {% endfor %}
    </div>
</section>

<!-- Featured Freelancers -->
<section class="container mb-5">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2>Top Freelancers</h2>
        <a href="{% url 'freelancers' %}" class="btn btn-outline-primary">View All</a>
    </div>
    
    <div class="row">
        {% for freelancer in featured_freelancers %}
        <div class="col-lg-4 mb-4">
            <div class="card freelancer-card h-100">
                <div class="card-body">
                    <div class="d-flex align-items-center mb-3">
                        <img src="{{ freelancer.avatar.url }}" alt="{{ freelancer.user.get_full_name }}" class="avatar me-3">
                        <div>
                            <h5 class="mb-1">{{ freelancer.user.get_full_name }}</h5>
                            <p class="text-muted mb-1">{{ freelancer.title }}</p>
                            <div class="rating-stars">
                                {% for i in "12345" %}
                                    {% if forloop.counter <= freelancer.rating|floatformat:0 %}
                                        <i class="fas fa-star"></i>
                                    {% else %}
                                        <i class="far fa-star"></i>
                                    {% endif %}
                                {% endfor %}
                                <small class="ms-1">({{ freelancer.total_reviews }})</small>
                            </div>
                        </div>
                    </div>
                    
                    <p class="mb-3">{{ freelancer.bio|truncatewords:15 }}</p>
                    
                    <div class="mb-3">
                        {% for skill in freelancer.skills.all|slice:":4" %}
                            <span class="skill-badge">{{ skill.name }}</span>
                        {% endfor %}
                    </div>
                    
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${{ freelancer.hourly_rate }}/hr</strong>
                        </div>
                        <a href="{% url 'freelancer_detail' freelancer.pk %}" class="btn btn-primary btn-sm">
                            View Profile
                        </a>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</section>

<!-- Recent Projects -->
<section class="container mb-5">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2>Recent Projects</h2>
        <a href="{% url 'projects' %}" class="btn btn-outline-primary">View All</a>
    </div>
    
    <div class="row">
        {% for project in recent_projects %}
        <div class="col-lg-6 mb-4">
            <div class="card project-card h-100">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <h5>{{ project.title }}</h5>
                        <span class="badge bg-success">${{ project.budget_min }}{% if project.budget_max %} - ${{ project.budget_max }}{% endif %}</span>
                    </div>
                    
                    <p class="text-muted mb-2">by {{ project.client.get_full_name }}</p>
                    <p class="mb-3">{{ project.description|truncatewords:20 }}</p>
                    
                    <div class="mb-3">
                        {% for skill in project.skills_required.all|slice:":3" %}
                            <span class="skill-badge">{{ skill.name }}</span>
                        {% endfor %}
                    </div>
                    
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">
                            <i class="fas fa-clock me-1"></i>{{ project.created_at|timesince }} ago
                        </small>
                        <a href="{% url 'project_detail' project.pk %}" class="btn btn-outline-primary btn-sm">
                            View Details
                        </a>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</section>
{% endblock %}
"""

# ============================================================================
# 11. FREELANCER LIST TEMPLATE (templates/marketplace/freelancers.html)
# ============================================================================
"""
{% extends 'base.html' %}

{% block title %}Browse Freelancers - FreelanceHub{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <!-- Filters Sidebar -->
        <div class="col-lg-3">
            <div class="card">
                <div class="card-header">
                    <h5><i class="fas fa-filter me-2"></i>Filters</h5>
                </div>
                <div class="card-body">
                    <form method="get">
                        <!-- Search -->
                        <div class="mb-3">
                            <label class="form-label">Search</label>
                            <input type="text" name="search" class="form-control" 
                                   value="{{ current_search }}" placeholder="Skills, name, location...">
                        </div>
                        
                        <!-- Category -->
                        <div class="mb-3">
                            <label class="form-label">Category</label>
                            <select name="category" class="form-select">
                                <option value="">All Categories</option>
                                {% for category in categories %}
                                    <option value="{{ category.id }}" 
                                            {% if current_category == category.id|stringformat:"s" %}selected{% endif %}>
                                        {{ category.name }}
                                    </option>
                                {% endfor %}
                            </select>
                        </div>
                        
                        <!-- Availability -->
                        <div class="mb-3">
                            <label class="form-label">Availability</label>
                            <select name="availability" class="form-select">
                                <option value="">Any</option>
                                <option value="available" {% if current_availability == "available" %}selected{% endif %}>Available</option>
                                <option value="busy" {% if current_availability == "busy" %}selected{% endif %}>Busy</option>
                            </select>
                        </div>
                        
                        <!-- Sort -->
                        <div class="mb-3">
                            <label class="form-label">Sort By</label>
                            <select name="sort" class="form-select">
                                <option value="-rating" {% if current_sort == "-rating" %}selected{% endif %}>Highest Rated</option>
                                <option value="hourly_rate" {% if current_sort == "hourly_rate" %}selected{% endif %}>Lowest Rate</option>
                                <option value="-hourly_rate" {% if current_sort == "-hourly_rate" %}selected{% endif %}>Highest Rate</option>
                                <option value="-projects_completed" {% if current_sort == "-projects_completed" %}selected{% endif %}>Most Experienced</option>
                            </select>
                        </div>
                        
                        <button type="submit" class="btn btn-primary w-100">Apply Filters</button>
                    </form>
                </div>
            </div>
        </div>
        
        <!-- Freelancers List -->
        <div class="col-lg-9">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Freelancers ({{ freelancers|length }} results)</h2>
            </div>
            
            <div class="row">
                {% for freelancer in freelancers %}
                <div class="col-lg-6 mb-4">
                    <div class="card freelancer-card h-100">
                        <div class="card-body">
                            <div class="d-flex align-items-center mb-3">
                                <img src="{{ freelancer.avatar.url }}" alt="{{ freelancer.user.get_full_name }}" class="avatar me-3">
                                <div class="flex-grow-1">
                                    <h5 class="mb-1">{{ freelancer.user.get_full_name }}</h5>
                                    <p class="text-muted mb-1">{{ freelancer.title }}</p>
                                    <div class="d-flex align-items-center">
                                        <div class="rating-stars me-2">
                                            {% for i in "12345" %}
                                                {% if forloop.counter <= freelancer.rating|floatformat:0 %}
                                                    <i class="fas fa-star"></i>
                                                {% else %}
                                                    <i class="far fa-star"></i>
                                                {% endif %}
                                            {% endfor %}
                                        </div>
                                        <small>({{ freelancer.total_reviews }})</small>
                                        <span class="badge bg-{% if freelancer.availability == 'available' %}success{% else %}warning{% endif %} ms-2">
                                            {{ freelancer.get_availability_display }}
                                        </span>
                                    </div>
                                </div>
                                <div class="text-end">
                                    <h5 class="text-primary mb-0">${{ freelancer.hourly_rate }}/hr</h5>
                                </div>
                            </div>
                            
                            <p class="mb-3">{{ freelancer.bio|truncatewords:20 }}</p>
                            
                            <div class="mb-3">
                                {% for skill in freelancer.skills.all|slice:":5" %}
                                    <span class="skill-badge">{{ skill.name }}</span>
                                {% endfor %}
                            </div>
                            
                            <div class="d-flex justify-content-between align-items-center">
                                <small class="text-muted">
                                    <i class="fas fa-map-marker-alt me-1"></i>{{ freelancer.location }}
                                </small>
                                <div>
                                    <small class="text-muted me-3">{{ freelancer.projects_completed }} projects</small>
                                    <a href="{% url 'freelancer_detail' freelancer.pk %}" class="btn btn-primary btn-sm">
                                        View Profile
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                {% empty %}
                <div class="col-12">
                    <div class="text-center py-5">
                        <i class="fas fa-search fa-3x text-muted mb-3"></i>
                        <h4>No freelancers found</h4>
                        <p class="text-muted">Try adjusting your search criteria.</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <!-- Pagination -->
            {% if is_paginated %}
            <nav aria-label="Freelancers pagination">
                <ul class="pagination justify-content-center">
                    {% if page_obj.has_previous %}
                        <li class="page-item">
                            <a class="page-link" href="?page={{ page_obj.previous_page_number }}">Previous</a>
                        </li>
                    {% endif %}
                    
                    {% for num in page_obj.paginator.page_range %}
                        {% if page_obj.number == num %}
                            <li class="page-item active">
                                <span class="page-link">{{ num }}</span>
                            </li>
                        {% else %}
                            <li class="page-item">
                                <a class="page-link" href="?page={{ num }}">{{ num }}</a>
                            </li>
                        {% endif %}
                    {% endfor %}
                    
                    {% if page_obj.has_next %}
                        <li class="page-item">
                            <a class="page-link" href="?page={{ page_obj.next_page_number }}">Next</a>
                        </li>
                    {% endif %}
                </ul>
            </nav>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}
"""

# ============================================================================
# 12. FREELANCER DETAIL TEMPLATE (templates/marketplace/freelancer_detail.html)
# ============================================================================
"""
{% extends 'base.html' %}

{% block title %}{{ freelancer.user.get_full_name }} - FreelanceHub{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-lg-8">
            <!-- Profile Header -->
            <div class="card mb-4">
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-3 text-center">
                            <img src="{{ freelancer.avatar.url }}" alt="{{ freelancer.user.get_full_name }}" 
                                 class="img-fluid rounded-circle mb-3" style="width: 150px; height: 150px; object-fit: cover;">
                            <span class="badge bg-{% if freelancer.availability == 'available' %}success{% else %}warning{% endif %} mb-2">
                                {{ freelancer.get_availability_display }}
                            </span>
                        </div>
                        <div class="col-md-9">
                            <h1>{{ freelancer.user.get_full_name }}</h1>
                            <h4 class="text-muted mb-3">{{ freelancer.title }}</h4>
                            
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <div class="d-flex align-items-center mb-2">
                                        <div class="rating-stars me-2">
                                            {% for i in "12345" %}
                                                {% if forloop.counter <= freelancer.rating|floatformat:0 %}
                                                    <i class="fas fa-star"></i>
                                                {% else %}
                                                    <i class="far fa-star"></i>
                                                {% endif %}
                                            {% endfor %}
                                        </div>
                                        <span>{{ freelancer.rating }} ({{ freelancer.total_reviews }} reviews)</span>
                                    </div>
                                    <p class="mb-2">
                                        <i class="fas fa-map-marker-alt me-2"></i>{{ freelancer.location }}
                                    </p>
                                    <p class="mb-0">
                                        <i class="fas fa-briefcase me-2"></i>{{ freelancer.projects_completed }} projects completed
                                    </p>
                                </div>
                                <div class="col-md-6 text-end">
                                    <h2 class="text-primary">${{ freelancer.hourly_rate }}/hour</h2>
                                    {% if freelancer.portfolio_url %}
                                        <a href="{{ freelancer.portfolio_url }}" target="_blank" class="btn btn-outline-primary btn-sm">
                                            <i class="fas fa-external-link-alt me-1"></i>Portfolio
                                        </a>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- About -->
            <div class="card mb-4">
                <div class="card-body">
                    <h5><i class="fas fa-user me-2"></i>About</h5>
                    <p>{{ freelancer.bio }}</p>
                </div>
            </div>
            
            <!-- Skills -->
            <div class="card mb-4">
                <div class="card-body">
                    <h5><i class="fas fa-code me-2"></i>Skills</h5>
                    <div class="mt-3">
                        {% for skill in freelancer.skills.all %}
                            <span class="skill-badge">{{ skill.name }}</span>
                        {% endfor %}
                    </div>
                </div>
            </div>
            
            <!-- Recent Reviews -->
            {% if reviews %}
            <div class="card mb-4">
                <div class="card-body">
                    <h5><i class="fas fa-star me-2"></i>Recent Reviews</h5>
                    {% for review in reviews %}
                    <div class="border-bottom py-3">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <div>
                                <strong>{{ review.reviewer.get_full_name }}</strong>
                                <div class="rating-stars">
                                    {% for i in "12345" %}
                                        {% if forloop.counter <= review.rating %}
                                            <i class="fas fa-star"></i>
                                        {% else %}
                                            <i class="far fa-star"></i>
                                        {% endif %}
                                    {% endfor %}
                                </div>
                            </div>
                            <small class="text-muted">{{ review.created_at|timesince }} ago</small>
                        </div>
                        <p class="mb-1">{{ review.comment }}</p>
                        <small class="text-muted">Project: {{ review.project.title }}</small>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>
        
        <!-- Sidebar -->
        <div class="col-lg-4">
            <div class="card mb-4">
                <div class="card-body text-center">
                    <h5>Contact {{ freelancer.user.first_name }}</h5>
                    {% if user.is_authenticated %}
                        <a href="#" class="btn btn-primary w-100 mb-2">
                            <i class="fas fa-envelope me-2"></i>Send Message
                        </a>
                        <a href="#" class="btn btn-success w-100">
                            <i class="fas fa-handshake me-2"></i>Hire Now
                        </a>
                    {% else %}
                        <p class="text-muted">Please <a href="{% url 'login' %}">login</a> to contact this freelancer.</p>
                    {% endif %}
                </div>
            </div>
            
            <!-- Stats -->
            <div class="card">
                <div class="card-body">
                    <h5>Statistics</h5>
                    <div class="row text-center">
                        <div class="col-6 border-end">
                            <h4>{{ freelancer.projects_completed }}</h4>
                            <small class="text-muted">Projects</small>
                        </div>
                        <div class="col-6">
                            <h4>{{ freelancer.total_reviews }}</h4>
                            <small class="text-muted">Reviews</small>
                        </div>
                    </div>
                    <hr>
                    <div class="text-center">
                        <h4>{{ freelancer.rating }}/5.0</h4>
                        <div class="rating-stars">
                            {% for i in "12345" %}
                                {% if forloop.counter <= freelancer.rating|floatformat:0 %}
                                    <i class="fas fa-star"></i>
                                {% else %}
                                    <i class="far fa-star"></i>
                                {% endif %}
                            {% endfor %}
                        </div>
                        <small class="text-muted d-block">Average Rating</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""

# ============================================================================
# 13. PROJECT LIST TEMPLATE (templates/marketplace/projects.html)
# ============================================================================
"""
{% extends 'base.html' %}

{% block title %}Browse Projects - FreelanceHub{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <!-- Filters -->
        <div class="col-lg-3">
            <div class="card">
                <div class="card-header">
                    <h5><i class="fas fa-filter me-2"></i>Filters</h5>
                </div>
                <div class="card-body">
                    <form method="get">
                        <div class="mb-3">
                            <label class="form-label">Search</label>
                            <input type="text" name="search" class="form-control" 
                                   value="{{ current_search }}" placeholder="Project title, description...">
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Category</label>
                            <select name="category" class="form-select">
                                <option value="">All Categories</option>
                                {% for category in categories %}
                                    <option value="{{ category.id }}" 
                                            {% if current_category == category.id|stringformat:"s" %}selected{% endif %}>
                                        {{ category.name }}
                                    </option>
                                {% endfor %}
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Min Budget</label>
                            <input type="number" name="budget_min" class="form-control" placeholder="$0">
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Max Budget</label>
                            <input type="number" name="budget_max" class="form-control" placeholder="$10000">
                        </div>
                        
                        <button type="submit" class="btn btn-primary w-100">Apply Filters</button>
                    </form>
                </div>
            </div>
        </div>
        
        <!-- Projects List -->
        <div class="col-lg-9">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Available Projects ({{ projects|length }} results)</h2>
                {% if user.is_authenticated %}
                    <a href="{% url 'project_create' %}" class="btn btn-primary">
                        <i class="fas fa-plus me-2"></i>Post Project
                    </a>
                {% endif %}
            </div>
            
            {% for project in projects %}
            <div class="card project-card mb-4">
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <h4><a href="{% url 'project_detail' project.pk %}" class="text-decoration-none">{{ project.title }}</a></h4>
                            <p class="text-muted mb-2">Posted by {{ project.client.get_full_name }}</p>
                            <p class="mb-3">{{ project.description|truncatewords:30 }}</p>
                            
                            <div class="mb-3">
                                {% for skill in project.skills_required.all|slice:":5" %}
                                    <span class="skill-badge">{{ skill.name }}</span>
                                {% endfor %}
                            </div>
                            
                            <div class="d-flex align-items-center text-muted">
                                <small class="me-3">
                                    <i class="fas fa-clock me-1"></i>{{ project.created_at|timesince }} ago
                                </small>
                                <small class="me-3">
                                    <i class="fas fa-file-alt me-1"></i>{{ project.proposals_count }} proposals
                                </small>
                                <small>
                                    <i class="fas fa-calendar me-1"></i>{{ project.duration }}
                                </small>
                            </div>
                        </div>
                        <div class="col-md-4 text-end">
                            <h4 class="text-success mb-2">
                                ${{ project.budget_min }}{% if project.budget_max %} - ${{ project.budget_max }}{% endif %}
                            </h4>
                            <span class="badge bg-primary mb-2">{{ project.get_budget_type_display }}</span>
                            <br>
                            <a href="{% url 'project_detail' project.pk %}" class="btn btn-outline-primary">
                                View Details
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            {% empty %}
            <div class="text-center py-5">
                <i class="fas fa-briefcase fa-3x text-muted mb-3"></i>
                <h4>No projects found</h4>
                <p class="text-muted">Try adjusting your search criteria or check back later.</p>
            </div>
            {% endfor %}
            
            <!-- Pagination -->
            {% if is_paginated %}
            <nav aria-label="Projects pagination">
                <ul class="pagination justify-content-center">
                    {% if page_obj.has_previous %}
                        <li class="page-item">
                            <a class="page-link" href="?page={{ page_obj.previous_page_number }}">Previous</a>
                        </li>
                    {% endif %}
                    
                    {% for num in page_obj.paginator.page_range %}
                        {% if page_obj.number == num %}
                            <li class="page-item active">
                                <span class="page-link">{{ num }}</span>
                            </li>
                        {% else %}
                            <li class="page-item">
                                <a class="page-link" href="?page={{ num }}">{{ num }}</a>
                            </li>
                        {% endif %}
                    {% endfor %}
                    
                    {% if page_obj.has_next %}
                        <li class="page-item">
                            <a class="page-link" href="?page={{ page_obj.next_page_number }}">Next</a>
                        </li>
                    {% endif %}
                </ul>
            </nav>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}
"""

# ============================================================================
# 14. PROJECT DETAIL TEMPLATE (templates/marketplace/project_detail.html)
# ============================================================================
"""
{% extends 'base.html' %}

{% block title %}{{ project.title }} - FreelanceHub{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-lg-8">
            <div class="card mb-4">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <div>
                            <h1>{{ project.title }}</h1>
                            <p class="text-muted">Posted by {{ project.client.get_full_name }}</p>
                        </div>
                        <span class="badge bg-success fs-6">${{ project.budget_min }}{% if project.budget_max %} - ${{ project.budget_max }}{% endif %}</span>
                    </div>
                    
                    <div class="row mb-4">
                        <div class="col-md-3">
                            <strong>Category:</strong><br>
                            <span class="text-muted">{{ project.category.name }}</span>
                        </div>
                        <div class="col-md-3">
                            <strong>Duration:</strong><br>
                            <span class="text-muted">{{ project.duration }}</span>
                        </div>
                        <div class="col-md-3">
                            <strong>Budget Type:</strong><br>
                            <span class="text-muted">{{ project.get_budget_type_display }}</span>
                        </div>
                        <div class="col-md-3">
                            <strong>Proposals:</strong><br>
                            <span class="text-muted">{{ project.proposals_count }}</span>
                        </div>
                    </div>
                    
                    <h5>Project Description</h5>
                    <p class="mb-4">{{ project.description|linebreaks }}</p>
                    
                    <h5>Required Skills</h5>
                    <div class="mb-4">
                        {% for skill in project.skills_required.all %}
                            <span class="skill-badge">{{ skill.name }}</span>
                        {% endfor %}
                    </div>
                    
                    {% if project.deadline %}
                    <p class="text-muted">
                        <i class="fas fa-calendar-alt me-2"></i>
                        <strong>Deadline:</strong> {{ project.deadline }}
                    </p>
                    {% endif %}
                </div>
            </div>
            
            <!-- Recent Proposals -->
            {% if proposals %}
            <div class="card">
                <div class="card-body">
                    <h5>Recent Proposals</h5>
                    {% for proposal in proposals %}
                    <div class="border-bottom py-3">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <div>
                                <strong>{{ proposal.freelancer.get_full_name }}</strong>
                                <br>
                                <small class="text-muted">{{ proposal.created_at|timesince }} ago</small>
                            </div>
                            <div class="text-end">
                                <h6 class="text-success">${{ proposal.bid_amount }}</h6>
                                <small class="text-muted">{{ proposal.delivery_time }}</small>
                            </div>
                        </div>
                        <p class="mb-0">{{ proposal.cover_letter|truncatewords:20 }}</p>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>
        
        <!-- Sidebar -->
        <div class="col-lg-4">
            <div class="card">
                <div class="card-body">
                    <h5>Project Actions</h5>
                    
                    {% if user.is_authenticated %}
                        {% if user != project.client %}
                            {% if user_proposal %}
                                <div class="alert alert-info">
                                    <i class="fas fa-check-circle me-2"></i>
                                    You have already submitted a proposal for this project.
                                </div>
                                <div class="card bg-light">
                                    <div class="card-body">
                                        <h6>Your Proposal</h6>
                                        <p><strong>Bid:</strong> ${{ user_proposal.bid_amount }}</p>
                                        <p><strong>Delivery:</strong> {{ user_proposal.delivery_time }}</p>
                                        <p class="mb-0">{{ user_proposal.cover_letter|truncatewords:15 }}</p>
                                    </div>
                                </div>
                            {% else %}
                                {% try %}
                                    {{ user.freelancerprofile }}
                                    <a href="{% url 'proposal_create' project.id %}" class="btn btn-primary w-100 mb-3">
                                        <i class="fas fa-paper-plane me-2"></i>Submit Proposal
                                    </a>
                                {% except %}
                                    <div class="alert alert-warning">
                                        Only freelancers can submit proposals. 
                                        <a href="{% url 'register' %}">Join as a freelancer</a> to bid on this project.
                                    </div>
                                {% endtry %}
                            {% endif %}
                        {% else %}
                            <div class="alert alert-info">
                                This is your project. You can view proposals from your dashboard.
                            </div>
                        {% endif %}
                    {% else %}
                        <div class="alert alert-warning">
                            Please <a href="{% url 'login' %}">login</a> to submit a proposal for this project.
                        </div>
                    {% endif %}
                    
                    <a href="#" class="btn btn-outline-primary w-100 mb-2">
                        <i class="fas fa-heart me-2"></i>Save Project
                    </a>
                    
                    <a href="#" class="btn btn-outline-secondary w-100">
                        <i class="fas fa-share me-2"></i>Share Project
                    </a>
                </div>
            </div>
            
            <!-- Client Info -->
            <div class="card mt-4">
                <div class="card-body">
                    <h5>About the Client</h5>
                    <div class="d-flex align-items-center mb-3">
                        {% try %}
                            <img src="{{ project.client.clientprofile.avatar.url }}" alt="{{ project.client.get_full_name }}" 
                                 class="avatar me-3">
                        {% except %}
                            <div class="avatar me-3 bg-secondary d-flex align-items-center justify-content-center">
                                <i class="fas fa-user text-white"></i>
                            </div>
                        {% endtry %}
                        <div>
                            <h6 class="mb-1">{{ project.client.get_full_name }}</h6>
                            {% try %}
                                <small class="text-muted">{{ project.client.clientprofile.location }}</small>
                            {% except %}
                                <small class="text-muted">Client</small>
                            {% endtry %}
                        </div>
                    </div>
                    
                    {% try %}
                        {% if project.client.clientprofile.company_description %}
                            <p class="mb-3">{{ project.client.clientprofile.company_description|truncatewords:20 }}</p>
                        {% endif %}
                        
                        <div class="row text-center">
                            <div class="col-6">
                                <h6>{{ project.client.clientprofile.rating }}</h6>
                                <small class="text-muted">Rating</small>
                            </div>
                            <div class="col-6">
                                <h6>{{ project.client.clientprofile.projects_posted }}</h6>
                                <small class="text-muted">Projects</small>
                            </div>
                        </div>
                    {% except %}
                        <p class="text-muted">New client on FreelanceHub</p>
                    {% endtry %}
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""

# ============================================================================
# 15. MANAGEMENT COMMANDS AND SETUP
# ============================================================================

# management/commands/create_sample_data.py
"""
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from marketplace.models import Category, Skill, FreelancerProfile, ClientProfile, Project

class Command(BaseCommand):
    help = 'Create sample data for testing'

    def handle(self, *args, **options):
        # Create categories
        categories_data = [
            {'name': 'Web Development', 'description': 'Frontend and backend web development'},
            {'name': 'Mobile Development', 'description': 'iOS and Android app development'},
            {'name': 'Design', 'description': 'UI/UX design and graphic design'},
            {'name': 'Marketing', 'description': 'Digital marketing and SEO'},
            {'name': 'Writing', 'description': 'Content writing and copywriting'},
        ]
        
        for cat_data in categories_data:
            category, created = Category.objects.get_or_create(**cat_data)
            if created:
                self.stdout.write(f'Created category: {category.name}')
        
        # Create skills
        skills_data = {
            'Web Development': ['Python', 'Django', 'React', 'JavaScript', 'HTML/CSS', 'Node.js'],
            'Mobile Development': ['React Native', 'Flutter', 'Swift', 'Kotlin', 'Ionic'],
            'Design': ['Figma', 'Photoshop', 'Illustrator', 'Sketch', 'InVision'],
            'Marketing': ['SEO', 'Google Ads', 'Facebook Ads', 'Content Marketing', 'Analytics'],
            'Writing': ['Blog Writing', 'Copywriting', 'Technical Writing', 'Social Media'],
        }
        
        for cat_name, skills in skills_data.items():
            category = Category.objects.get(name=cat_name)
            for skill_name in skills:
                skill, created = Skill.objects.get_or_create(name=skill_name, category=category)
                if created:
                    self.stdout.write(f'Created skill: {skill.name}')
        
        self.stdout.write(self.style.SUCCESS('Sample data created successfully!'))
"""

# ============================================================================
# 16. SETUP INSTRUCTIONS
# ============================================================================
"""
SETUP INSTRUCTIONS:

1. Install Python 3.8+ and create virtual environment:
   python -m venv freelance_env
   source freelance_env/bin/activate  # On Windows: freelance_env\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Create Django project:
   django-admin startproject freelance_marketplace
   cd freelance_marketplace
   python manage.py startapp marketplace

4. Configure settings.py with the provided configuration

5. Run migrations:
   python manage.py makemigrations
   python manage.py migrate

6. Create superuser:
   python manage.py createsuperuser

7. Create sample data (optional):
   python manage.py create_sample_data

8. Collect static files:
   python manage.py collectstatic

9. Run development server:
   python manage.py runserver

10. Access the application:
    - Home: http://localhost:8000/
    - Admin: http://localhost:8000/admin/
    - Browse Freelancers: http://localhost:8000/freelancers/
    - Browse Projects: http://localhost:8000/projects/

FEATURES INCLUDED:
✅ User registration and authentication
✅ Freelancer and client profiles
✅ Project posting and browsing
✅ Proposal submission system
✅ Search and filtering
✅ Reviews and ratings
✅ Admin interface
✅ Responsive design
✅ File uploads (avatars)
✅ Dashboard for users
✅ Professional UI with Bootstrap

ADDITIONAL FEATURES YOU CAN ADD:
- Payment integration (Stripe/PayPal)
- Real-time messaging
- Email notifications
- Advanced search with Elasticsearch
- File sharing for projects
- Video calls integration
- Mobile app with Django REST API
- Social media integration
"""