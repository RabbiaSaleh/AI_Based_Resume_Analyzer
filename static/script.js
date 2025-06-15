document.addEventListener('DOMContentLoaded', function() {
    // Drag & Drop logic for upload form
    const dropArea = document.getElementById('upload-form');
    const fileElem = document.getElementById('fileElem');
    const fileLabel = document.getElementById('fileLabel');
    const fileList = document.getElementById('fileList');
    const browseBtn = fileLabel ? fileLabel.querySelector('.browse-btn') : null;
    const uploadStatus = document.getElementById('uploadStatus');

    function showFileName() {
        fileList.innerHTML = '';
        if (fileElem.files.length > 0) {
            const file = fileElem.files[0];
            const div = document.createElement('div');
            div.className = 'file-item';
            div.style = "margin-bottom:0.5rem; color:#3949ab; font-size:1.1rem; font-weight:600;";
            div.textContent = file.name;
            fileList.appendChild(div);
            if (uploadStatus) {
                uploadStatus.textContent = "CV is uploaded";
                uploadStatus.style.color = "#43a047";
            }
        } else {
            if (uploadStatus) uploadStatus.textContent = "";
        }
    }

    if (dropArea && fileElem && fileLabel && fileList && browseBtn) {
        // Highlight drop area
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.add('dragover');
            }, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.remove('dragover');
            }, false);
        });

        // Handle drop
        dropArea.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileElem.files = e.dataTransfer.files;
                showFileName();
            }
        });

        // Show file name when selected
        fileElem.addEventListener('change', showFileName);

        // Clicking "Upload" text triggers file input
        browseBtn.addEventListener('click', function(e) {
            e.preventDefault();
            fileElem.click();
        });
    }

    // Modal logic
    function showModal(id) {
        var modal = document.getElementById(id);
        if (modal) modal.style.display = 'flex';
    }
    function hideModal(id) {
        var modal = document.getElementById(id);
        if (modal) modal.style.display = 'none';
    }

    var loginBtn = document.getElementById('loginBtn');
    var signupBtn = document.getElementById('signupBtn');
    var closeLogin = document.getElementById('closeLogin');
    var closeSignup = document.getElementById('closeSignup');
    var switchToLogin = document.getElementById('switchToLogin');

    if (loginBtn) {
        loginBtn.onclick = function(e) {
            e.preventDefault();
            showModal('loginModal');
        };
    }
    if (signupBtn) {
        signupBtn.onclick = function(e) {
            e.preventDefault();
            showModal('signupModal');
        };
    }
    if (closeLogin) {
        closeLogin.onclick = function() { hideModal('loginModal'); };
    }
    if (closeSignup) {
        closeSignup.onclick = function() { hideModal('signupModal'); };
    }
    if (switchToLogin) {
        switchToLogin.onclick = function(e) {
            e.preventDefault();
            hideModal('signupModal');
            showModal('loginModal');
        };
    }

    // Hide modal on outside click
    window.onclick = function(event) {
        var loginModal = document.getElementById('loginModal');
        var signupModal = document.getElementById('signupModal');
        if (event.target === loginModal) loginModal.style.display = "none";
        if (event.target === signupModal) signupModal.style.display = "none";
    };

    // Password validation for signup
    var signupForm = document.getElementById('signupForm');
    var passwordInput = document.getElementById('signup-password');
    var confirmInput = document.getElementById('signup-confirm');
    var passwordReq = document.getElementById('passwordReq');

    function validatePassword(pw) {
        // ≤ 8 chars, at least 1 special char, at least 1 number
        return pw && pw.length <= 8 && /[!@#$%^&*(),.?":{}|<>]/.test(pw) && /\d/.test(pw);
    }

    if (signupForm && passwordInput && confirmInput && passwordReq) {
        signupForm.addEventListener('submit', function(e) {
            var pw = passwordInput.value;
            var cpw = confirmInput.value;
            if (!validatePassword(pw)) {
                passwordReq.style.color = "#ffb300";
                e.preventDefault();
            } else if (pw !== cpw) {
                passwordReq.textContent = "Passwords do not match.";
                passwordReq.style.color = "#ffb300";
                e.preventDefault();
            } else {
                passwordReq.style.color = "#fffde7";
                passwordReq.textContent = "Password must be ≤ 8 characters, include a number and a special character.";
            }
        });

        passwordInput.addEventListener('input', function() {
            if (validatePassword(passwordInput.value)) {
                passwordReq.style.color = "#b2ff59";
                passwordReq.textContent = "Password looks good!";
            } else {
                passwordReq.style.color = "#fffde7";
                passwordReq.textContent = "Password must be ≤ 8 characters, include a number and a special character.";
            }
        });
    }

    // Mobile menu toggle
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navContainer = document.querySelector('nav .container');
    if (mobileMenuBtn && navContainer) {
        mobileMenuBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            navContainer.classList.toggle('mobile-nav-active');
            document.body.style.overflow = navContainer.classList.contains('mobile-nav-active') ? 'hidden' : '';
        });
        // Close menu when clicking a link (for better UX)
        const navLinks = navContainer.querySelectorAll('.nav-links a');
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                navContainer.classList.remove('mobile-nav-active');
                document.body.style.overflow = '';
            });
        });
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (
                navContainer.classList.contains('mobile-nav-active') &&
                !navContainer.contains(e.target) &&
                e.target !== mobileMenuBtn
            ) {
                navContainer.classList.remove('mobile-nav-active');
                document.body.style.overflow = '';
            }
        });
    }

    // Listen for message from results.html to open login/signup modal
    window.addEventListener('message', function(event) {
        if (event.data && event.data.action === "showLoginOrSignupModal") {
            // Prefer signup modal, fallback to login modal
            var signupModal = document.getElementById('signupModal');
            var loginModal = document.getElementById('loginModal');
            if (signupModal) {
                signupModal.style.display = 'flex';
                var input = signupModal.querySelector('input[type="email"],input[type="text"]');
                if (input) setTimeout(function(){ input.focus(); }, 100);
            } else if (loginModal) {
                loginModal.style.display = 'flex';
                var input = loginModal.querySelector('input[type="email"],input[type="text"]');
                if (input) setTimeout(function(){ input.focus(); }, 100);
            }
        }
    });
});