BRANCH=$(git branch | grep \* | cut -d ' ' -f2)

git pull origin master

# git checkout master
# git pull
# git checkout $BRANCH
# git merge origin master

if [ -z "$1" ];
then
  hub pull-request
  exit
fi

if [ ! "$1" = "--no-edit" ];
then
  hub pull-request -m "$BRANCH: $1" -m "$2" -b master
else
  hub pull-request -b master --no-edit
fi