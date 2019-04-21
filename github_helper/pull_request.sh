BRANCH=$(git branch | grep \* | cut -d ' ' -f2)

git checkout master
git pull
git checkout $BRANCH
git merge origin master
if [ "$1" = "edit" ];
then
  hub pull-request -m "$BRANCH: " -b master -e
else
  hub pull-request -b master --no-edit
fi