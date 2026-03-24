### Environment Overview
- **User Name**: Xiaoming
- **User Background**: A music enthusiast who enjoys playing songs based on mood.

### Entities in the Environment
#### Entity: Song
- Description: A track entry in the music collection.
- Attributes:
  - **Title**: The name of the song.
  - **Rating**: The user's rating for the song.
- Available Operations:
  - **create**: Create a new instance of this entity.
  - **read**: Retrieve one or more attribute values of this entity.
  - **update**: Modify one or more attribute values of this entity.
  - **delete**: Remove an instance of this entity.
  - **play**: Play this song.

#### Entity: Account
- Description: The user's personal account.
- Attributes:
  - **Name**: The name of the account.
  - **Balance**: The current balance of the account.
- Available Operations:
  - **create**: Create a new instance of this entity.
  - **read**: Retrieve one or more attribute values of this entity.
  - **update**: Modify one or more attribute values of this entity.
  - **delete**: Remove an instance of this entity.

### Task Preferences
The task should involve the following characteristics:
- **Average number of entities involved**: 2
- **Average number of operations involved**: 2
- **Relation difficulty**: Hard: Involves multiple entities or attributes, and operations require prior condition checks or depend on the results of previous steps. Requires reasoning and decision-making.
