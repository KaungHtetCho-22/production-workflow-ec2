FROM mysql:latest

HEALTHCHECK --interval=30s --timeout=3s \
  CMD mysqladmin ping -h localhost -u root --password=$MYSQL_ROOT_PASSWORD || exit 1

# Copy initialization scripts
# COPY ./init.sql /docker-entrypoint-initdb.d/

EXPOSE 3306
VOLUME /var/lib/mysql