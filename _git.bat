git init
git add .
git commit -m "Primeiro commit - adicionando código Python"
git remote add origin https://github.com/costadiegus/serper_ollama_crewai.git
git push -u origin main
git add .github/workflows/ci.yml
git commit -m "Adiciona configuração do CI com Poetry"
git push