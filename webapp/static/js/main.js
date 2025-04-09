// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 高亮当前活动菜单项
    highlightActiveMenuItem();
    
    // 监听排序控件的改变
    setupSortControls();
});

// 高亮当前页面对应的菜单项
function highlightActiveMenuItem() {
    const currentPath = window.location.pathname;
    const menuItems = document.querySelectorAll('.menu a');
    
    menuItems.forEach(item => {
        if (item.getAttribute('href') === currentPath) {
            item.parentElement.classList.add('active');
        } else {
            item.parentElement.classList.remove('active');
        }
    });
}

// 设置排序控件
function setupSortControls() {
    const sortBySelect = document.getElementById('sort-by');
    const sortOrderSelect = document.getElementById('sort-order');
    
    if (sortBySelect && sortOrderSelect) {
        // 当排序选项改变时提交表单
        sortBySelect.addEventListener('change', function() {
            document.getElementById('sort-form').submit();
        });
        
        sortOrderSelect.addEventListener('change', function() {
            document.getElementById('sort-form').submit();
        });
    }
}

// 文件上传验证
function validateFileUpload() {
    const fileInput = document.getElementById('pcap-file');
    if (!fileInput.files.length) {
        alert('请选择要上传的文件');
        return false;
    }
    
    const fileName = fileInput.files[0].name;
    const fileExt = fileName.split('.').pop().toLowerCase();
    
    if (fileExt !== 'pcap') {
        alert('只支持上传.pcap格式的文件');
        return false;
    }
    
    // 显示加载指示器
    document.getElementById('loading-indicator').style.display = 'block';
    return true;
} 