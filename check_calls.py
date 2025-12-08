import os
import re

# 搜索整个目录下的Python文件
def search_filter_image_calls(directory):
    calls_found = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 搜索调用模式，如 await _filter_image(, await self._filter_image(, await self.image_processor_service._filter_image( 等
                        call_patterns = [
                            r'await\s+_filter_image\(',
                            r'await\s+self\._filter_image\(',
                            r'await\s+self\.image_processor_service\._filter_image\(',
                            r'await\s+[^.]+\._filter_image\('
                        ]
                        
                        for pattern in call_patterns:
                            matches = re.finditer(pattern, content)
                            for match in matches:
                                # 获取匹配行
                                lines = content[:match.start()].split('\n')
                                line_number = len(lines) + 1
                                calls_found.append((file_path, line_number, match.group()))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return calls_found

# 搜索定义
def search_filter_image_definitions(directory):
    definitions = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 搜索定义模式
                        def_pattern = r'def\s+_filter_image\(|async\s+def\s+_filter_image\('
                        matches = re.finditer(def_pattern, content)
                        for match in matches:
                            lines = content[:match.start()].split('\n')
                            line_number = len(lines) + 1
                            definitions.append((file_path, line_number, match.group()))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return definitions

# 执行搜索
directory = '.'
calls = search_filter_image_calls(directory)
definitions = search_filter_image_definitions(directory)

print("_filter_image方法定义:")
for file_path, line, definition in definitions:
    print(f"{file_path}:{line} - {definition}")

print("\n_filter_image方法调用:")
if calls:
    for file_path, line, call in calls:
        print(f"{file_path}:{line} - {call}")
else:
    print("没有找到_filter_image方法的调用！")
