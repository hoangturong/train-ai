# English

## API Documentation

## Introduction

This documentation provides details on the endpoints and usage of the API. The API allows you to perform operations related to a game and retrieve game data.

## Base URL

    https://api-taixiu.ndng.net/

## Endpoints

### GET /game/:number

This endpoint generates a game result by simulating dice rolls.

#### Parameters

- **number** (required): The number of rolls to simulate.

#### Response

The response will contain the following properties:

- **response**: The sum of the random numbers generated.
- **data**: An array of individual random numbers generated.
- **type**: The type of the game result, either "Tài" or "Xỉu" based on the sum being even or odd.

Example Response:

```json
{
	"response": 15,
	"data": [3, 2, 4, 1, 5],
	"type": "Tài"
}
```

### GET /data/:number

This endpoint retrieves the game history data.

#### Parameters

- **number** (required): The number of recent game results to retrieve.

#### Response

The response will be an array containing the specified number of recent game results.

Example Response:

```json
[
	{
		"response": 15,
		"data": [3, 2, 4, 1, 5],
		"type": "Tài"
	},
	{
		"response": 10,
		"data": [2, 1, 3, 2, 2],
		"type": "Xỉu"
	}
]
```

## Error Handling

The API may return the following error responses:

- **400 Bad Request**: When an invalid or missing parameter is provided. Example Response:

```json
{
	"message": "Please enter a valid number"
}
```

- **500 Internal Server Error**: When there is an error reading or writing the data file. Example Response:

```json
{
	"message": "Failed to read/write data file"
}
```

## Conclusion

This concludes the documentation for the API. Feel free to explore and use the available endpoints. If you have any questions or issues, please contact the API developers.
