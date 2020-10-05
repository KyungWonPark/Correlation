#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <errno.h>

// Macros
#define TESTING_CHECK( err ) 												 \
	do { 																	 \
		magma_int_t err_ = ( err ); 										 \
		if ( err_ != 0 ) { 													 \
			fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
					 #err, __FILE__, __LINE__, 								 \
					 (long long) err_, magma_strerror(err_) ); 				 \
			exit(1); 														 \
		} 																	 \
	} while( 0 ) 															 \

// ./testing_custom <job_size> <mat_buffer_shmID> <eigVal_shmID> <eigVec_shmID>
int main(int argc, char* argv[]) {
	// Parse arguments
	int jobSize = 13362;
	char* fileName = argv[1];

	// MAGMA Init
	TESTING_CHECK( magma_init() );

	// DSYEVD routine parameters 
	magma_int_t ngpu = 4;
	magma_vec_t jobz = MagmaVec;
	magma_uplo_t uplo = MagmaUpper;
	magma_int_t N = (magma_int_t) jobSize;
	double* h_R;
	magma_int_t lda = N;
	double* w1;
	double* h_work;
	magma_int_t lwork;
	magma_int_t* iwork;
	magma_int_t liwork;
	magma_int_t info;

	// Query parameters
	double aux_work[1];
	magma_int_t aux_iwork[1];

	// Query for workspace sizes
	magma_dsyevd(jobz, uplo, N, NULL, lda, NULL, aux_work, -1, aux_iwork, -1, &info);
	
	lwork = (magma_int_t) aux_work[0];
	liwork = aux_iwork[0];

	// Allocate host memory for workload
	TESTING_CHECK( magma_dmalloc_pinned( &h_R, N*lda ));
	TESTING_CHECK( magma_dmalloc_cpu( &w1, N));
	TESTING_CHECK( magma_dmalloc_pinned( &h_work, lwork ));
	TESTING_CHECK( magma_imalloc_cpu( &iwork, liwork ));

	// Copy matrix into MAGMA pinned memory
	lapackf77_dlacpy( MagmaFullStr, &N, &N, pMatBuffer, &lda, h_R, &lda);
	CsvParser* csvparser = CsvParser_new(fileName, ",", 0);
	CsvRow *row;

	int rowIdx = 0;
	while ((row = Csvparser_getRow(csvparser))) {
		const char **rowFields = CsvParser_getFields(row);
		for (int i = 0; i < CsvParser_getNumFields(row); i++) {
			double d;
			sscanf(rowFields[i], "%lf", &d);
			h_R[rowIdx * 13362 + i] = d;
		}
	}	

	// Carry out calculations
	magma_dsyevd_m(ngpu, jobz, uplo, N, h_R, lda, w1, h_work, lwork, iwork, liwork, &info);
	if (info != 0) {
		printf("lapackf77_dsyevd returned error: %s.\n", magma_strerror( info ));
		exit(1);
	}

	// Finish
	magma_free_cpu( w1 );
	magma_free_cpu( iwork );
	magma_free_pinned( h_R );
	magma_free_pinned( h_work );

	TESTING_CHECK( magma_finalize() );
	return 0;
}

// CsvParser
#ifdef __cplusplus
extern "C" {
#endif

CsvParser *CsvParser_new(const char *filePath, const char *delimiter, int firstLineIsHeader) {
    CsvParser *csvParser = (CsvParser*)malloc(sizeof(CsvParser));
    if (filePath == NULL) {
        csvParser->filePath_ = NULL;
    } else {
        int filePathLen = strlen(filePath);
        csvParser->filePath_ = (char*)malloc((filePathLen + 1));
        strcpy(csvParser->filePath_, filePath);
    }
    csvParser->firstLineIsHeader_ = firstLineIsHeader;
    csvParser->errMsg_ = NULL;
    if (delimiter == NULL) {
        csvParser->delimiter_ = ',';
    } else if (_CsvParser_delimiterIsAccepted(delimiter)) {
        csvParser->delimiter_ = *delimiter;
    } else {
        csvParser->delimiter_ = '\0';
    }
    csvParser->header_ = NULL;
    csvParser->fileHandler_ = NULL;
        csvParser->fromString_ = 0;
        csvParser->csvString_ = NULL;
        csvParser->csvStringIter_ = 0;

    return csvParser;
}

CsvParser *CsvParser_new_from_string(const char *csvString, const char *delimiter, int firstLineIsHeader) {
        CsvParser *csvParser = CsvParser_new(NULL, delimiter, firstLineIsHeader);
        csvParser->fromString_ = 1;
        if (csvString != NULL) {
                int csvStringLen = strlen(csvString);
                csvParser->csvString_ = (char*)malloc(csvStringLen + 1);
                strcpy(csvParser->csvString_, csvString);
        }
        return csvParser;
}

void CsvParser_destroy(CsvParser *csvParser) {
    if (csvParser == NULL) {
        return;
    }
    if (csvParser->filePath_ != NULL) {
        free(csvParser->filePath_);
    }
    if (csvParser->errMsg_ != NULL) {
        free(csvParser->errMsg_);
    }
    if (csvParser->fileHandler_ != NULL) {
        fclose(csvParser->fileHandler_);
    }
    if (csvParser->header_ != NULL) {
        CsvParser_destroy_row(csvParser->header_);
    }
        if (csvParser->csvString_ != NULL) {
                free(csvParser->csvString_);
        }
    free(csvParser);
}

void CsvParser_destroy_row(CsvRow *csvRow) {
    int i;
    for (i = 0 ; i < csvRow->numOfFields_ ; i++) {
        free(csvRow->fields_[i]);
    }
        free(csvRow->fields_);
    free(csvRow);
}

const CsvRow *CsvParser_getHeader(CsvParser *csvParser) {
    if (! csvParser->firstLineIsHeader_) {
        _CsvParser_setErrorMessage(csvParser, "Cannot supply header, as current CsvParser object does not support header");
        return NULL;
    }
    if (csvParser->header_ == NULL) {
        csvParser->header_ = _CsvParser_getRow(csvParser);
    }
    return csvParser->header_;
}

CsvRow *CsvParser_getRow(CsvParser *csvParser) {
    if (csvParser->firstLineIsHeader_ && csvParser->header_ == NULL) {
        csvParser->header_ = _CsvParser_getRow(csvParser);
    }
    return _CsvParser_getRow(csvParser);
}

int CsvParser_getNumFields(const CsvRow *csvRow) {
    return csvRow->numOfFields_;
}

const char **CsvParser_getFields(const CsvRow *csvRow) {
    return (const char**)csvRow->fields_;
}
CsvRow *_CsvParser_getRow(CsvParser *csvParser) {
    int numRowRealloc = 0;
    int acceptedFields = 64;
    int acceptedCharsInField = 64;
    if (csvParser->filePath_ == NULL && (! csvParser->fromString_)) {
        _CsvParser_setErrorMessage(csvParser, "Supplied CSV file path is NULL");
        return NULL;
    }
    if (csvParser->csvString_ == NULL && csvParser->fromString_) {
        _CsvParser_setErrorMessage(csvParser, "Supplied CSV string is NULL");
        return NULL;
    }
    if (csvParser->delimiter_ == '\0') {
        _CsvParser_setErrorMessage(csvParser, "Supplied delimiter is not supported");
        return NULL;
    }
    if (! csvParser->fromString_) {
        if (csvParser->fileHandler_ == NULL) {
            csvParser->fileHandler_ = fopen(csvParser->filePath_, "r");
            if (csvParser->fileHandler_ == NULL) {
                int errorNum = errno;
                const char *errStr = strerror(errorNum);
                char *errMsg = (char*)malloc(1024 + strlen(errStr));
                strcpy(errMsg, "");
                sprintf(errMsg, "Error opening CSV file for reading: %s : %s", csvParser->filePath_, errStr);
                _CsvParser_setErrorMessage(csvParser, errMsg);
                free(errMsg);
                return NULL;
            }
        }
    }
CsvRow *csvRow = (CsvRow*)malloc(sizeof(CsvRow));
    csvRow->fields_ = (char**)malloc(acceptedFields * sizeof(char*));
    csvRow->numOfFields_ = 0;
    int fieldIter = 0;
    char *currField = (char*)malloc(acceptedCharsInField);
    int inside_complex_field = 0;
    int currFieldCharIter = 0;
    int seriesOfQuotesLength = 0;
    int lastCharIsQuote = 0;
    int isEndOfFile = 0;
    while (1) {
        char currChar = (csvParser->fromString_) ? csvParser->csvString_[csvParser->csvStringIter_] : fgetc(csvParser->fileHandler_);
        csvParser->csvStringIter_++;
        int endOfFileIndicator;
        if (csvParser->fromString_) {
            endOfFileIndicator = (currChar == '\0');
        } else {
            endOfFileIndicator = feof(csvParser->fileHandler_);
        }
        if (endOfFileIndicator) {
            if (currFieldCharIter == 0 && fieldIter == 0) {
                _CsvParser_setErrorMessage(csvParser, "Reached EOF");
                                free(currField);
                                CsvParser_destroy_row(csvRow);
                return NULL;
            }
            currChar = '\n';
            isEndOfFile = 1;
        }
        if (currChar == '\r') {
            continue;
        }
if (currFieldCharIter == 0  && ! lastCharIsQuote) {
            if (currChar == '\"') {
                inside_complex_field = 1;
                lastCharIsQuote = 1;
                continue;
            }
        } else if (currChar == '\"') {
            seriesOfQuotesLength++;
            inside_complex_field = (seriesOfQuotesLength % 2 == 0);
            if (inside_complex_field) {
                currFieldCharIter--;
            }
        } else {
            seriesOfQuotesLength = 0;
        }
        if (isEndOfFile || ((currChar == csvParser->delimiter_ || currChar == '\n') && ! inside_complex_field) ){
            currField[lastCharIsQuote ? currFieldCharIter - 1 : currFieldCharIter] = '\0';
            csvRow->fields_[fieldIter] = (char*)malloc(currFieldCharIter + 1);
            strcpy(csvRow->fields_[fieldIter], currField);
            free(currField);
            csvRow->numOfFields_++;
            if (currChar == '\n') {
                return csvRow;
            }
            if (csvRow->numOfFields_ != 0 && csvRow->numOfFields_ % acceptedFields == 0) {
                csvRow->fields_ = (char**)realloc(csvRow->fields_, ((numRowRealloc + 2) * acceptedFields) * sizeof(char*));
                numRowRealloc++;
            }
            acceptedCharsInField = 64;
            currField = (char*)malloc(acceptedCharsInField);
            currFieldCharIter = 0;
            fieldIter++;
            inside_complex_field = 0;
			} else {
            currField[currFieldCharIter] = currChar;
            currFieldCharIter++;
            if (currFieldCharIter == acceptedCharsInField - 1) {
                acceptedCharsInField *= 2;
                currField = (char*)realloc(currField, acceptedCharsInField);
            }
        }
        lastCharIsQuote = (currChar == '\"') ? 1 : 0;
    }
}

int _CsvParser_delimiterIsAccepted(const char *delimiter) {
    char actualDelimiter = *delimiter;
    if (actualDelimiter == '\n' || actualDelimiter == '\r' || actualDelimiter == '\0' ||
            actualDelimiter == '\"') {
        return 0;
    }
    return 1;
}

void _CsvParser_setErrorMessage(CsvParser *csvParser, const char *errorMessage) {
    if (csvParser->errMsg_ != NULL) {
        free(csvParser->errMsg_);
    }
    int errMsgLen = strlen(errorMessage);
    csvParser->errMsg_ = (char*)malloc(errMsgLen + 1);
    strcpy(csvParser->errMsg_, errorMessage);
}

const char *CsvParser_getErrorMessage(CsvParser *csvParser) {
    return csvParser->errMsg_;
}

#ifdef __cplusplus
}
#endif

// CsvWriter

#ifdef __cplusplus
extern "C" {
#endif

CsvWriter *CsvWriter_new(const char *filePath, const char *delimiter, int append) {
        CsvWriter *csvWriter = malloc(sizeof(CsvWriter));
        if (filePath == NULL) {
        csvWriter->filePath_ = NULL;
    } else {
        int filePathLen = strlen(filePath);
        csvWriter->filePath_ = malloc(filePathLen + 1);
        strcpy(csvWriter->filePath_, filePath);
    }
        csvWriter->rowIsNew_ = 1;
        csvWriter->append_ = append;
        csvWriter->fileHandler_ = NULL;
        csvWriter->errMsg_ = NULL;
         if (delimiter == NULL) {
        csvWriter->delimiter_ = ',';
    } else if (_CsvWriter_delimiterIsAccepted(delimiter)) {
        csvWriter->delimiter_ = *delimiter;
    } else {
        csvWriter->delimiter_ = '\0';
    }

        return csvWriter;
}
void CsvWriter_destroy(CsvWriter *csvWriter) {
        if (csvWriter == NULL) {
                return;
        }
        if (csvWriter->filePath_ != NULL) {
                free(csvWriter->filePath_);
        }
        if (csvWriter->fileHandler_ != NULL) {
                fclose(csvWriter->fileHandler_);
        }
        free(csvWriter);
}

int CsvWriter_nextRow(CsvWriter *csvWriter) {
        if (csvWriter->filePath_ != NULL && _CsvWriter_ensureFileIsOpen(csvWriter)) {
                return 1;
        }
        if (csvWriter->fileHandler_ != NULL) {
                fprintf(csvWriter->fileHandler_, "\n");
        } else {
                printf("\n");
        }
        csvWriter->rowIsNew_ = 1;

        return 0;
}

int CsvWriter_writeField(CsvWriter *csvWriter, char *field) {
        if (csvWriter->delimiter_ == '\0') {
                _CsvWriter_setErrorMessage(csvWriter, "Supplied delimiter is not supported");
                return 1;
        }
        if (csvWriter->filePath_ != NULL && _CsvWriter_ensureFileIsOpen(csvWriter)) {
                return 1;
        }
        if (field == NULL) {
                _CsvWriter_setErrorMessage(csvWriter, "NULL string was passed");
                return 1;
        }
        char *fieldPrefix = csvWriter->rowIsNew_ ? "" : ",";
        int complexField = (strchr(field, csvWriter->delimiter_) || strchr(field, '\n') || strchr(field, '\"')) ? 1 : 0;
        if (! complexField) {
                if (csvWriter->fileHandler_ != NULL) {
                        fprintf(csvWriter->fileHandler_, "%s%s", fieldPrefix, field);
                } else {
                        printf("%s%s", fieldPrefix, field);
                }
                csvWriter->rowIsNew_ = 0;
                return 0;
        }
        char buffer[1024];
        strcpy(buffer, "");
        int bufferIter = 0;
        int fieldIter;
        int repeatedOnQuotes = 0;
        int bufferWasWrittenOnce = 0;
        int bufferWasJustFlushed = 0;
        int repeatSchedulerIndex = -1;
        for (fieldIter = 0 ; field[fieldIter] != '\0' ; fieldIter++) {
                bufferWasJustFlushed = 0;
                buffer[bufferIter] = field[fieldIter];
                if (bufferIter == 1022) {
                        buffer[1023] = '\0';
                        if (! bufferWasWrittenOnce) {
                                if (csvWriter->fileHandler_ != NULL) {
                                        fprintf(csvWriter->fileHandler_, "%s\"%s", fieldPrefix, buffer);
                                } else {
                                        printf("%s\"%s", fieldPrefix, buffer);
								}
                                bufferWasWrittenOnce = 1;
                        } else {
                                if (csvWriter->fileHandler_ != NULL) {
                                        fprintf(csvWriter->fileHandler_, "%s", buffer);
                                } else {
                                        printf("%s", buffer);
                                }
                        }
                        strcpy(buffer, "");
                        bufferIter = -1;
                        bufferWasJustFlushed = 1;
                }
                if (field[fieldIter] == '\"' && ! repeatedOnQuotes) {
                        repeatSchedulerIndex = fieldIter;
                        fieldIter--;
                        repeatedOnQuotes = 1;
                }
                if (repeatSchedulerIndex == fieldIter) {
                        repeatedOnQuotes = 0;
                }
                bufferIter++;
        }
        if (! bufferWasJustFlushed) {
                buffer[bufferIter] = '\0';
                if (! bufferWasWrittenOnce) {
                        if (csvWriter->fileHandler_ != NULL) {
                                fprintf(csvWriter->fileHandler_, "%s\"%s", fieldPrefix, buffer);
                        } else {
                                printf("%s\"%s", fieldPrefix, buffer);
                        }
                        bufferWasWrittenOnce = 1;
                } else {
                        if (csvWriter->fileHandler_ != NULL) {
                                fprintf(csvWriter->fileHandler_, "%s", buffer);
                        } else {
                                printf("%s", buffer);
                        }
                }
        }
		if (csvWriter->fileHandler_ != NULL) {
                fprintf(csvWriter->fileHandler_, "\"");
        } else {
                printf("\"");
        }

        csvWriter->rowIsNew_ = 0;
        return 0;
}


const char *CsvWriter_getErrorMessage(CsvWriter *csvWriter) {
        return csvWriter->errMsg_;
}

int _CsvWriter_delimiterIsAccepted(const char *delimiter) {
    char actualDelimiter = *delimiter;
    if (actualDelimiter == '\n' || actualDelimiter == '\r' || actualDelimiter == '\0' || actualDelimiter == '\"') {
        return 0;
    }
    return 1;
}

int _CsvWriter_ensureFileIsOpen(CsvWriter *csvWriter) {
        if (csvWriter->filePath_ == NULL) {
                _CsvWriter_setErrorMessage(csvWriter, "Supplied CSV file path is NULL");
                return 1;
        }
        if (csvWriter->fileHandler_ != NULL) {
                return 0;
        }
        char *openType = csvWriter->append_ ? "a" : "w";
        csvWriter->fileHandler_ = fopen(csvWriter->filePath_, openType);
        if (csvWriter->fileHandler_ == NULL) {
                int errorNum = errno;
                const char *errStr = strerror(errorNum);
                char *errMsg = malloc(1024 + strlen(errStr));
                strcpy(errMsg, "");
                sprintf(errMsg, "Error opening CSV file for writing/appending: %s : %s", csvWriter->filePath_, errStr);
                _CsvWriter_setErrorMessage(csvWriter, errMsg);
                free(errMsg);
                return 1;
        }
        return 0;
}

void _CsvWriter_setErrorMessage(CsvWriter *csvWriter, const char *errorMessage) {
    if (csvWriter->errMsg_ != NULL) {
        free(csvWriter->errMsg_);
    }
    int errMsgLen = strlen(errorMessage);
    csvWriter->errMsg_ = malloc(errMsgLen + 1);
    strcpy(csvWriter->errMsg_, errorMessage);
}

#ifdef __cplusplus
}
#endif

