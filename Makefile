up:
	docker compose up -d --build

stop:
	docker compose down

restart:
	docker compose down
	docker compose up -d --build

back-logs:
	docker compose logs -f term_epidemic_api

migrations:
	docker compose exec term_epidemic_api sh -lc "python manage.py makemigrations && python manage.py migrate"
