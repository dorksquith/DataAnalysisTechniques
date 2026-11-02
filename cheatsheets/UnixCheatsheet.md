---
kernelspec:
  name: python3
  display_name: 'Python 3'
numbering:
  title: false
  headings: false
authors:
  - name: Tom Stevenson
---

(chapter:unixcs)=
# Unix Cheatsheet

Some basic commands to get you started with a Unix based operating system.

More will be added as we use them throughout the course.

:::{tip}
Don't forget to press ⏻ then ▷ as per the [intro](#chapter:intro)
:::

:::{tip}
Anytime a command has `<something>` replace with the appropriate input (including the `< >`)
:::

## A.1 Navigation & File Operations

:::{list-table} Commands for navigating the directory structure in the terminal and performing operations on files
:label: unixcsTab1
:header-rows: 1
:enumerator: A.1

* - Command
  - Description
* - `ls`
  - List all files in current directory
* - `pwd`
  - Print current working directory
* - `cd </path/to/dir>`
  - Change directory (`cd ..` to go up, `cd ~` to go to home)
* - `cp <src> <dest>`
  - Copy file from source to destination (use `cp -r` to recursively copy)
* - `mv <old_name> <new_name>`
  - Move or rename files and directories
* - `rm <name>`
  - Removes file with specified name
* - `rm -r <name>`
  - Removes recursively e.g. for directories and files within. Be careful!
* - `touch <filename>`
  - Create an empty file or update timestamps
* - `man <program>`
  - Displays built-in documentation (manual) for program
:::

## A.2 Viewing Files

:::{list-table} Commands for viewing files in the terminal
:label: unixcsTab2
:align: center
:header-rows: 1
:enumerator: A.2

* - Command
  - Description
* - `cat <filename>`
  - Output entire file content to terminal
* - `less <filename>`
  - View file interactively with scrolling (`q` to quit)
* - `head -n 20 <filename>`
  - View the first 20 lines of a file
* - `tail -n 20 <filename>`
  - View the last 20 lines of a file
* - `wc -l <filename>`
  - Count the total lines in a file
:::