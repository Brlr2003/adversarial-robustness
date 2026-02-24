% Epoka University MSc Thesis LaTeX Class
% Version 1.0 - January 2026
% Based on Epoka University Thesis Writing Guidelines
%
% This class file provides formatting for MSc theses at Epoka University
% according to the official guidelines.

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{epoka}[2026/01/08 v1.0 Epoka University Thesis Class]

% Base class
\LoadClass[a4paper,oneside,onecolumn,12pt]{report}[1996/10/31]

% Required packages
\RequirePackage{ifthen}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{fancyhdr}
\RequirePackage[hang,labelsep=period]{caption}
\RequirePackage{calc}
\RequirePackage{setspace}
\RequirePackage{geometry}
\RequirePackage{tocloft}
\RequirePackage{titlesec}
\RequirePackage{indentfirst}
\RequirePackage{mathptmx}  % Times New Roman font

% Caption setup for tables and figures
\captionsetup[table]{position=top, aboveskip=10pt, belowskip=10pt}
\captionsetup[figure]{position=bottom, aboveskip=10pt, belowskip=10pt}

% Page geometry - Epoka University requirements
% Margins: Left 4cm, Right 2.5cm, Top 2.5cm, Bottom 2.5cm
\geometry{
    a4paper,
    left=4cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm,
    headheight=15pt,
    headsep=10pt,
    footskip=1.25cm % Modified to move footer UP (approx 1.25cm from bottom)
}

% Header and footer setup - page numbers bottom centered
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}  % Page number at bottom center
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Line spacing
\onehalfspacing

% Paragraph formatting
\setlength{\parindent}{12mm}
\setlength{\parskip}{6pt}     % Default to 6pt (Abstract, Refs, Body) per PDF instructions

% Make figure and table captions bold
\captionsetup{labelfont=bf}

% Hyphenation penalties
\hyphenpenalty=5000
\exhyphenpenalty=5000

% Boolean flags
\newif\if@itsaMSc
\@itsaMSctrue  % Default is MSc

\newif\if@TabloListesiVar
\@TabloListesiVartrue

\newif\if@SekilListesiVar
\@SekilListesiVartrue

\newif\if@AbbreviationVar
\@AbbreviationVarfalse

\newif\if@DedicationVar
\@DedicationVarfalse

% Commands for thesis type
\newcommand*{\MSc}{\@itsaMSctrue}
\newcommand*{\PhD}{\@itsaMScfalse}

% Storage commands
\newcommand{\th@StoreIn}[2]{\gdef#1{#2}}

% Author information
\renewcommand*{\author}[2]{%
    \th@StoreIn{\th@firstname}{#1}%
    \th@StoreIn{\th@lastname}{#2}%
    \th@StoreIn{\th@fullname}{#1 #2}%
}

% Thesis information
\renewcommand{\title}[1]{\th@StoreIn{\th@title}{#1}}
\newcommand{\titleenglish}[1]{\th@StoreIn{\th@titleenglish}{#1}}
\newcommand{\titlealbanian}[1]{\th@StoreIn{\th@titlealbanian}{#1}}
\renewcommand{\date}[1]{\th@StoreIn{\th@date}{#1}}
\newcommand{\dateenglish}[1]{\th@StoreIn{\th@dateenglish}{#1}}

% Supervisor information
\newcommand{\supervisor}[1]{\th@StoreIn{\th@supervisor}{#1}}
\newcommand{\supervisorenglish}[1]{\th@StoreIn{\th@supervisorenglish}{#1}}

% Department information
\newcommand{\department}[1]{\th@StoreIn{\th@department}{#1}}
\newcommand{\departmentenglish}[1]{\th@StoreIn{\th@departmentenglish}{#1}}
\newcommand{\headofdepartment}[1]{\th@StoreIn{\th@headofdepartment}{#1}}

% Faculty information
\newcommand{\faculty}[1]{\th@StoreIn{\th@faculty}{#1}}
\newcommand{\facultyenglish}[1]{\th@StoreIn{\th@facultyenglish}{#1}}

% Student ID
\newcommand{\studentid}[1]{\th@StoreIn{\th@studentid}{#1}}

% Committee members
\newcommand{\committee}[5]{%
    \th@StoreIn{\th@committeeone}{#1}%
    \th@StoreIn{\th@committeetwo}{#2}%
    \th@StoreIn{\th@committeethree}{#3}%
    \th@StoreIn{\th@committeefour}{#4}%
    \th@StoreIn{\th@committeefive}{#5}%
}

% Front matter content
\renewcommand{\abstract}[1]{\th@StoreIn{\th@abstractcontent}{#1}}
\newcommand{\abstrakt}[1]{\th@StoreIn{\th@abstraktcontent}{#1}}
\newcommand{\acknowledgments}[1]{\th@StoreIn{\th@acknowledgments}{#1}}
\newcommand{\dedication}[1]{\@DedicationVartrue\th@StoreIn{\th@dedication}{#1}}
\newcommand{\abbreviations}[1]{\@AbbreviationVartrue\th@StoreIn{\th@abbreviations}{#1}}

% Keywords
\newcommand{\keywords}[1]{\th@StoreIn{\th@keywords}{#1}}
\newcommand{\keywordsenglish}[1]{\th@StoreIn{\th@keywordsenglish}{#1}}

% List control
\newcommand*{\NoTableList}{\@TabloListesiVarfalse}
\newcommand*{\NoFigureList}{\@SekilListesiVarfalse}

% Chapter formatting - 16pt font size
\titleformat{\chapter}[display]
    {\normalfont\bfseries\centering\fontsize{16}{19.2}\selectfont}
    {\MakeUppercase{\chaptertitlename}\ \thechapter}
    {20pt}
    {\bfseries\fontsize{16}{19.2}\selectfont\MakeUppercase}

\titlespacing*{\chapter}{0pt}{12pt}{12pt} % 12pt before and after chapter title

% Section formatting - with indentation
\titleformat{\section}
    {\normalfont\fontsize{14}{16.8}\selectfont\bfseries}
    {\thesection}
    {1em}
    {}
[\vspace{-0.5em}]  % Reduce space after section title

\titlespacing*{\section}{0.7cm}{12pt}{12pt}  % Left indent 7mm, 12pt before/after

% Subsection formatting - with indentation
\titleformat{\subsection}
    {\normalfont\fontsize{13}{15.6}\selectfont\bfseries}
    {\thesubsection}
    {1em}
    {}

\titlespacing*{\subsection}{0.7cm}{6pt}{6pt}  % Left indent 7mm to align with Section

% Subsubsection formatting - with indentation
\titleformat{\subsubsection}
    {\normalfont\bfseries}
    {\thesubsubsection}
    {1em}
    {}

\titlespacing*{\subsubsection}{0.7cm}{6pt}{6pt}  % Left indent 7mm to align with Section

% Table of Contents formatting - 16pt centered
\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\renewcommand{\listtablename}{LIST OF TABLES}
\renewcommand{\bibname}{REFERENCES}

% TOC depth
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

% Standard TOC Entry formatting for chapters
\renewcommand{\cftchapfont}{\bfseries}
\renewcommand{\cftchappagefont}{\bfseries}
\renewcommand{\cftdot}{.}
\renewcommand{\cftdotsep}{1} % Denser dots (matching template)
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftchapnumwidth}{0pt}
\setlength{\cftbeforechapskip}{12pt} 

% Section and Subsection TOC formatting
\setlength{\cftsecindent}{1.25cm}
\setlength{\cftsecnumwidth}{1.5em}
\setlength{\cftbeforesecskip}{8pt} % Spacing for 1.5 lines feel
\setlength{\cftsubsecindent}{2.0cm}
\setlength{\cftsubsecnumwidth}{2.5em}
\setlength{\cftbeforesubsecskip}{8pt}
\setlength{\cftsubsubsecindent}{2.5cm}
\setlength{\cftsubsubsecnumwidth}{3em}
\setlength{\cftbeforesubsubsecskip}{8pt}

% Centering preliminary titles
\renewcommand{\cfttoctitlefont}{\hfill\fontsize{16}{19.2}\selectfont\bfseries}
\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftlottitlefont}{\hfill\fontsize{16}{19.2}\selectfont\bfseries}
\renewcommand{\cftafterlottitle}{\hfill}
\renewcommand{\cftloftitlefont}{\hfill\fontsize{16}{19.2}\selectfont\bfseries}
\renewcommand{\cftafterloftitle}{\hfill}

% Redefine internal chapter command to insert two lines in TOC correctly
\def\@chapter[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
        \refstepcounter{chapter}%
        \typeout{\@chapapp\space\thechapter.}%
        % Insert "CHAPTER X" line
        \addcontentsline{toc}{chapter}{CHAPTER \thechapter}%
        % Insert title line
        \addcontentsline{toc}{chapter}{#1}%
    \else
        \addcontentsline{toc}{chapter}{#1}%
    \fi
    \chaptermark{#1}%
    \addtocontents{lof}{\protect\addvspace{10\p@}}%
    \addtocontents{lot}{\protect\addvspace{10\p@}}%
    \if@twocolumn
        \@topnewpage[\@makechapterhead{#2}]%
    \else
        \@makechapterhead{#2}%
        \@afterheading
    \fi
}

% Figure and table numbering
\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}

% Title Page - Following Epoka University Format
\newcommand{\makecoverpage}{%
    \begin{titlepage}
        \centering
        \singlespacing
        
        % Title at the top
        {\normalsize \MakeUppercase{\th@title}\\[6pt]}
        
        % Empty line (1.5 spacing)
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Submission statement
        {\normalsize A THESIS SUBMITTED TO\\}
        {\normalsize THE \MakeUppercase{\th@faculty}\\}
        {\normalsize OF\\}
        {\normalsize EPOKA UNIVERSITY\\}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % BY
        {\normalsize BY\\}
        
        % Empty line
        \vspace{12pt}
        
        % Author name
        {\normalsize \MakeUppercase{\th@fullname}\\}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Degree statement
        {\normalsize IN PARTIAL FULFILLMENT OF THE REQUIREMENTS\\}
        {\normalsize FOR\\}
        \if@itsaMSc
            {\normalsize THE DEGREE OF MASTER OF SCIENCE\\}
        \else
            {\normalsize THE DEGREE OF DOCTOR OF PHILOSOPHY\\}
        \fi
        {\normalsize IN\\}
        {\normalsize \MakeUppercase{\th@department}\\}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Empty line
        \vspace{12pt}
        
        % Date - push to bottom with vfill
        \vfill
        {\normalsize \MakeUppercase{\th@date}}
        
        % Bottom margin
        \vspace{2cm}
        
    \end{titlepage}
    \clearpage
}

% Approval Page (Page i)
\newcommand{\makeapprovalpage}{%
    \thispagestyle{plain}
    \begin{center}
        \textbf{Approval sheet of the Thesis}\\[1cm]
    \end{center}
    
    \noindent
    This is to certify that we have read this thesis entitled ``\textbf{\MakeUppercase{\th@titleenglish}}'' and that in our opinion it is fully adequate, in scope and quality, as a thesis for the degree of \if@itsaMSc Master of Science\else Doctor of Philosophy\fi.
    
    \vspace{2cm}
    
    \begin{flushright}
        \rule{6cm}{0.4pt}\\
        \th@headofdepartment\\
        Head of Department\\
        Date: \rule{3cm}{0.4pt}
    \end{flushright}
    
    \vspace{1.5cm}
    
    \noindent
    Examining Committee Members:\par\vspace{0.5cm}
    
    \noindent
    \begin{tabular}{@{}p{0.45\textwidth}p{0.3\textwidth}l@{}}
        \th@committeeone & (\th@departmentenglish) & \rule{2.5cm}{0.4pt} \\[0.8cm]
        \th@committeetwo & (\th@departmentenglish) & \rule{2.5cm}{0.4pt} \\[0.8cm]
        \th@committeethree & (\th@departmentenglish) & \rule{2.5cm}{0.4pt} \\
    \end{tabular}
    
    \clearpage
}

% Declaration Page (Page ii)
\newcommand{\makedeclarationpage}{%
    \thispagestyle{plain}
    
    \noindent
    \textbf{I hereby declare that all information in this document has been obtained and presented in accordance with academic rules and ethical conduct. I also declare that, as required by these rules and conduct, I have fully cited and referenced all material and results that are not original to this work.}
    
    \vspace{3cm}
    
    \begin{flushright}
        \begin{tabular}{@{}rl@{}}
            Name Surname: & \th@fullname \\[0.5cm]
            Signature: & \rule{4cm}{0.4pt}
        \end{tabular}
    \end{flushright}
    
    \clearpage
}

% Abstract page
\newcommand{\makeabstractpage}{%
    \chapter*{ABSTRACT}
    \addcontentsline{toc}{chapter}{ABSTRACT}
    
    \begin{center}
        \vspace*{-12pt} % Adjusting for chapter header space
        \vspace{\baselineskip}
        {\bfseries\fontsize{14}{16.8}\selectfont \MakeUppercase{\th@titleenglish}}\\[\baselineskip]
        \vspace{\baselineskip}
        \th@lastname, \th@firstname\\
        \if@itsaMSc M.Sc.,\else Ph.D.,\fi\ Department of \th@departmentenglish\\
        Supervisor: \th@supervisorenglish\\[\baselineskip]
    \end{center}
    
    \th@abstractcontent
    
    \vspace{\baselineskip}
    \vspace{\baselineskip}
    \vspace{\baselineskip}
    
    \noindent
    \textbf{Keywords:} \textit{\th@keywordsenglish}
    
    \clearpage
}

% Albanian Abstract page
\newcommand{\makeabstraktpage}{%
    \chapter*{ABSTRAKT}
    \addcontentsline{toc}{chapter}{ABSTRAKT}
    
    \begin{center}
        \vspace*{-12pt}
        \vspace{\baselineskip}
        {\bfseries\fontsize{14}{16.8}\selectfont \MakeUppercase{\th@title}}\\[\baselineskip]
        \vspace{\baselineskip}
        \th@lastname, \th@firstname\\
        \if@itsaMSc M.Sc.,\else Ph.D.,\fi\ Departamenti i \th@department\\
        Mbikëqyrës: \th@supervisor\\[10pt]
    \end{center}
    
    \th@abstraktcontent
    
    \vspace{\baselineskip}
    \noindent
    \textbf{Fjalë kyçe:} \textit{\th@keywords}
    
    \clearpage
}

% Acknowledgments page
\newcommand{\makeacknowledgments}{%
    \chapter*{\fontsize{16}{19.2}\selectfont\bfseries ACKNOWLEDGMENTS}
    \addcontentsline{toc}{chapter}{ACKNOWLEDGMENTS}
    \vspace{20pt}
    
    \indent \th@acknowledgments
    
    \clearpage
}

% Dedication page (optional)
\newcommand{\makededication}{%
    \if@DedicationVar
        \thispagestyle{plain}
        \vspace*{5cm}
        
        \begin{center}
            \textit{\th@dedication}
        \end{center}
        
        \clearpage
    \fi
}

% Abbreviations page
\newcommand{\makeabbreviations}{%
    \if@AbbreviationVar
        \chapter*{LIST OF ABBREVIATIONS}
        \addcontentsline{toc}{chapter}{LIST OF ABBREVIATIONS}
        
        \th@abbreviations
        
        \clearpage
    \fi
}

% Front matter
\newcommand{\makefrontmatter}{%
    \pagenumbering{roman}
    \setcounter{page}{1}
    
    \makecoverpage
    \makeapprovalpage
    \makedeclarationpage
    \makeabstractpage
    \makeabstraktpage
    \makededication
    \makeacknowledgments
    \makeabbreviations
    
    % Table of Contents
    \begin{singlespace}
        \tableofcontents
    \end{singlespace}
    \clearpage
    
    % List of Tables (12pt spacing per guidelines)
    \if@TabloListesiVar
        {%
         \setlength{\parskip}{12pt}%
         \listoftables
        }
        \addcontentsline{toc}{chapter}{LIST OF TABLES}
        \clearpage
    \fi
    
    % List of Figures (12pt spacing per guidelines)
    \if@SekilListesiVar
        {%
         \setlength{\parskip}{12pt}%
         \listoffigures
        }
        \addcontentsline{toc}{chapter}{LIST OF FIGURES}
        \clearpage
    \fi
    
    % Start main matter
    \pagenumbering{arabic}
    \setcounter{page}{1}
}

% Bibliography setup
\bibliographystyle{plain}

\endinput
